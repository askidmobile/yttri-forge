//! Индекс «префикс диалога → страницы paged-KV» для переиспользования KV.
//!
//! Зачем: сейчас popaдание в prefix-cache стоит копии KV внимания host→device
//! (~1.6 ГБ на 24.5k токенов, ≈0.53 с из 0.78 с повторного префила). Страницы
//! paged-пула **неизменяемы после записи**: диалог только дописывает новые.
//! Значит вместо копирования состояния достаточно удержать страницы и на
//! попадании перепривязать block table (`paged_kv_cuda::stage_block_table_row`).
//!
//! Этот модуль — чистая структура данных (без CUDA), поэтому её можно
//! тестировать где угодно. Она отвечает на один вопрос: по токенам запроса
//! найти самый длинный уже посчитанный префикс и отдать его страницы по слоям.
//!
//! Спецификация: `docs/specs/2026-08-27-tiered-kv-cache.md` (сценарий 1).

use std::collections::{HashMap, VecDeque};

/// Размер страницы. Должен совпадать с `PAGE_SIZE` в `paged_kv_cuda.rs`.
pub const PAGE_TOKENS: usize = 64;

/// Отпечаток конфигурации: совпадения ищутся только внутри одной
/// (модель, квант, тип KV). Иначе чужие страницы были бы подставлены молча.
pub type Fingerprint = u64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageHit {
    /// Сколько токенов префикса покрыто (кратно PAGE_TOKENS).
    pub prefix_tokens: usize,
    /// Страницы по слоям: `pages[layer]` — физические page-id для этого слоя.
    pub pages: Vec<Vec<u32>>,
}

#[derive(Debug)]
struct Entry {
    id: u64,
    key: u64,
    fingerprint: Fingerprint,
    /// Токены записи — для защиты от коллизии хеша (сверяем, а не верим хешу).
    tokens: Vec<u32>,
    pages: Vec<Vec<u32>>,
    lru_index: (),
}

/// Цепочка хешей по полным страницам: h_n = hash(h_{n-1}, tokens[n*P..(n+1)*P]).
/// Неполная последняя страница не участвует — её содержимое зависит от длины
/// запроса и не может совпадать со следующей репликой.
pub fn chain_hashes(tokens: &[u32]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let n = tokens.len() / PAGE_TOKENS;
    let mut out = Vec::with_capacity(n);
    let mut prev: u64 = 0;
    for page in tokens[..n * PAGE_TOKENS].chunks(PAGE_TOKENS) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        prev.hash(&mut h);
        page.hash(&mut h);
        prev = h.finish();
        out.push(prev);
    }
    out
}

pub struct PagePrefixIndex {
    buckets: HashMap<u64, Vec<u64>>,
    by_id: HashMap<u64, Entry>,
    lru: VecDeque<u64>,
    /// Бюджет в страницах (суммарно по всем удерживаемым префиксам).
    budget_pages: usize,
    used_pages: usize,
    next_id: u64,
}

impl PagePrefixIndex {
    pub fn new(budget_pages: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            by_id: HashMap::new(),
            lru: VecDeque::new(),
            budget_pages,
            used_pages: 0,
            next_id: 1,
        }
    }

    pub fn used_pages(&self) -> usize {
        self.used_pages
    }

    pub fn entries(&self) -> usize {
        self.by_id.len()
    }

    /// Удержать страницы одного префикса. Пустой префикс (меньше страницы) не
    /// удерживается: попадание с него невозможно.
    ///
    /// Возвращает `true`, если запись сохранена (в том числе если такая уже
    /// была — тогда просто обновляется LRU).
    pub fn put(
        &mut self,
        fingerprint: Fingerprint,
        tokens: &[u32],
        pages: Vec<Vec<u32>>,
    ) -> bool {
        let n_pages = tokens.len() / PAGE_TOKENS;
        if n_pages == 0 || pages.is_empty() {
            return false;
        }
        let pages_total: usize = pages.iter().map(|p| p.len()).sum();
        if pages_total > self.budget_pages {
            return false;
        }
        let key = *chain_hashes(tokens).last().expect("n_pages > 0");
        let trimmed = tokens[..n_pages * PAGE_TOKENS].to_vec();
        // Дедуп: тот же префикс уже удержан — обновляем только LRU.
        if let Some(id) = self
            .buckets
            .get(&key)
            .and_then(|ids| {
                ids.iter().copied().find(|&id| {
                    self.by_id.get(&id).is_some_and(|e| {
                        e.fingerprint == fingerprint && e.tokens == trimmed
                    })
                })
            })
        {
            self.touch(id);
            return true;
        }
        while self.used_pages + pages_total > self.budget_pages {
            if !self.evict_one() {
                return false;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.used_pages += pages_total;
        self.lru.push_back(id);
        self.buckets.entry(key).or_default().push(id);
        self.by_id.insert(
            id,
            Entry {
                id,
                key,
                fingerprint,
                tokens: trimmed,
                pages,
                lru_index: (),
            },
        );
        true
    }

    /// Самый длинный удержанный префикс запроса внутри того же отпечатка.
    /// Возвращается `prefix_tokens < tokens.len()`: на досчёт хвоста нужен
    /// хотя бы один токен, чтобы получить логиты.
    pub fn find(&mut self, fingerprint: Fingerprint, tokens: &[u32]) -> Option<PageHit> {
        let hashes = chain_hashes(tokens);
        for &h in hashes.iter().rev() {
            let Some(ids) = self.buckets.get(&h).map(Vec::as_slice) else {
                continue;
            };
            let mut best: Option<u64> = None;
            let mut best_len = 0usize;
            for &id in ids {
                let Some(e) = self.by_id.get(&id) else { continue };
                if e.fingerprint != fingerprint {
                    continue;
                }
                if e.tokens.len() > best_len
                    && e.tokens.len() < tokens.len()
                    && tokens[..e.tokens.len()] == e.tokens[..]
                {
                    best = Some(id);
                    best_len = e.tokens.len();
                }
            }
            if let Some(id) = best {
                self.touch(id);
                let e = &self.by_id[&id];
                return Some(PageHit {
                    prefix_tokens: e.tokens.len(),
                    pages: e.pages.clone(),
                });
            }
        }
        None
    }

    fn touch(&mut self, id: u64) {
        if let Some(pos) = self.lru.iter().position(|&x| x == id) {
            self.lru.remove(pos);
        }
        self.lru.push_back(id);
    }

    /// Вытеснить самый старый префикс. Возвращает страницы, которые вызывающий
    /// обязан вернуть в free-list пула.
    pub fn evict_one(&mut self) -> bool {
        while let Some(id) = self.lru.pop_front() {
            if let Some(e) = self.by_id.remove(&id) {
                let n: usize = e.pages.iter().map(|p| p.len()).sum();
                self.used_pages = self.used_pages.saturating_sub(n);
                if let Some(ids) = self.buckets.get_mut(&e.key) {
                    ids.retain(|&x| x != id);
                    if ids.is_empty() {
                        self.buckets.remove(&e.key);
                    }
                }
                return true;
            }
        }
        false
    }

    /// Слить все удержанные страницы (вызывающему — вернуть их в пул).
    pub fn drain_pages(&mut self) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        for (_, e) in self.by_id.drain() {
            out.extend(e.pages);
        }
        self.buckets.clear();
        self.lru.clear();
        self.used_pages = 0;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    fn pages_for(tokens_len: usize, layers: usize) -> Vec<Vec<u32>> {
        let np = tokens_len / PAGE_TOKENS;
        (0..layers)
            .map(|l| (0..np as u32).map(|p| p + 1 + (l as u32) * 1000).collect())
            .collect()
    }

    #[test]
    fn exact_repeat_hits_and_leaves_one_token() {
        let mut idx = PagePrefixIndex::new(1024);
        let t = toks(PAGE_TOKENS * 4);
        assert!(idx.put(7, &t, pages_for(t.len(), 2)));
        // Запрос длиннее на токен: должен найти префикс из 4 страниц.
        let mut q = t.clone();
        q.push(999);
        let hit = idx.find(7, &q).expect("hit");
        assert_eq!(hit.prefix_tokens, PAGE_TOKENS * 4);
        assert_eq!(hit.pages.len(), 2);
        assert_eq!(hit.pages[0].len(), 4);
        // Точный повтор того же промпта попаданием не считается: нужен хвост.
        assert!(idx.find(7, &t).is_none());
    }

    #[test]
    fn extending_prompt_reuses_common_prefix() {
        let mut idx = PagePrefixIndex::new(1024);
        let base = toks(PAGE_TOKENS * 2);
        assert!(idx.put(1, &base, pages_for(base.len(), 1)));
        let mut next = base.clone();
        next.extend(toks(PAGE_TOKENS * 3));
        let hit = idx.find(1, &next).expect("hit");
        assert_eq!(hit.prefix_tokens, PAGE_TOKENS * 2);
    }

    #[test]
    fn divergent_tail_misses() {
        let mut idx = PagePrefixIndex::new(1024);
        let a = toks(PAGE_TOKENS * 3);
        assert!(idx.put(1, &a, pages_for(a.len(), 1)));
        let mut b = a.clone();
        let last = b.len() - 1;
        b[last] = 12345;
        assert!(idx.find(1, &b).is_none());
    }

    #[test]
    fn different_fingerprint_never_matches() {
        let mut idx = PagePrefixIndex::new(1024);
        let a = toks(PAGE_TOKENS * 3);
        assert!(idx.put(1, &a, pages_for(a.len(), 1)));
        let mut q = a.clone();
        q.push(5);
        assert!(idx.find(2, &q).is_none(), "чужая конфигурация не подставляется");
        assert!(idx.find(1, &q).is_some());
    }

    #[test]
    fn short_prompts_are_not_kept() {
        let mut idx = PagePrefixIndex::new(1024);
        let short = toks(PAGE_TOKENS - 1);
        assert!(!idx.put(1, &short, vec![vec![1, 2]]));
        assert_eq!(idx.entries(), 0);
    }

    #[test]
    fn budget_evicts_lru_and_frees_pages() {
        // Бюджет ровно на две записи по 2 страницы.
        let mut idx = PagePrefixIndex::new(4);
        let a = toks(PAGE_TOKENS * 2);
        let mut b = toks(PAGE_TOKENS * 2);
        b[0] = 777;
        assert!(idx.put(1, &a, pages_for(a.len(), 1)));
        assert!(idx.put(1, &b, pages_for(b.len(), 1)));
        assert_eq!(idx.used_pages(), 4);
        // Третья запись вытесняет самую старую (a).
        let mut c = toks(PAGE_TOKENS * 2);
        c[0] = 888;
        assert!(idx.put(1, &c, pages_for(c.len(), 1)));
        assert_eq!(idx.used_pages(), 4);
        assert_eq!(idx.entries(), 2);
        let mut qa = a.clone();
        qa.push(1);
        assert!(idx.find(1, &qa).is_none(), "вытесненная запись не находится");
        let mut qc = c.clone();
        qc.push(1);
        assert!(idx.find(1, &qc).is_some());
    }

    #[test]
    fn drain_returns_pages_and_empties() {
        let mut idx = PagePrefixIndex::new(64);
        let a = toks(PAGE_TOKENS * 3);
        assert!(idx.put(1, &a, pages_for(a.len(), 2)));
        let drained = idx.drain_pages();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].len(), 3);
        assert_eq!(idx.used_pages(), 0);
        assert_eq!(idx.entries(), 0);
    }

    #[test]
    fn duplicate_put_updates_lru_without_doubling() {
        let mut idx = PagePrefixIndex::new(64);
        let a = toks(PAGE_TOKENS * 2);
        assert!(idx.put(1, &a, pages_for(a.len(), 1)));
        assert!(idx.put(1, &a, pages_for(a.len(), 1)));
        assert_eq!(idx.entries(), 1);
        assert_eq!(idx.used_pages(), 2);
    }
}
