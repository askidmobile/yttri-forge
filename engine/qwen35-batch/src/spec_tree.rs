//! Trie-представление спекулятивного дерева.
//!
//! Это первый шаг к однопроходной проверке ветвящегося драфта:
//! узлы хранят токен, родителя и диапазон поддерева в линейном порядке.
//! `FlatTrie` используется для передачи дерева в CUDA и построения маски внимания.

#[derive(Debug, Clone, PartialEq)]
pub struct TrieNode {
    pub token: u32,
    pub parent: Option<usize>,
    pub logprob: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatTrieNode {
    pub node: TrieNode,
    pub start: usize,
    pub end: usize,
    pub depth: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlatTrie {
    nodes: Vec<FlatTrieNode>,
}

/// Кандидат ветвящегося драфта.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TreeDraftCandidate {
    pub token: u32,
    pub logprob: f32,
}

/// Параметры построения дерева.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TreeDraftSpec {
    /// Общее число узлов, включая root-sentinel.
    pub budget: usize,
    pub depth: usize,
    pub branching: usize,
}

/// Узлы для CUDA-стейджинга tree-verify.
///
/// Порядок — линейный порядок `FlatTrie`: одна строка запроса на узел.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeStaging {
    pub tokens: Vec<u32>,
    pub parents: Vec<usize>,
    pub depths: Vec<u32>,
    pub starts: Vec<u32>,
    pub ends: Vec<u32>,
}

/// Проверить, видит ли строка `row` ключ `key` в host-маске tree-attention.
///
/// `prefix_len` — длина committed-префикса; узел дерева занимает суффикс после него.
/// Это host-эталон для CUDA-маски: полезен в тестах и CPU-fallback.
pub fn trie_mask_allows(staging: &TreeStaging, prefix_len: usize, row: usize, key: usize) -> bool {
    if row >= staging.depths.len() {
        return false;
    }
    if key < prefix_len {
        return true;
    }
    let key_node = key - prefix_len;
    if key_node >= staging.depths.len() {
        return false;
    }
    // Диапазон принадлежит КЛЮЧУ: строка `row` видит ключ `key_node`,
    // если она лежит в поддереве этого ключа. Так root виден всем строкам,
    // а брат/сестра — только строкам своей ветки.
    let start = staging.starts[key_node] as usize;
    let end = staging.ends[key_node] as usize;
    row >= start && row <= end
}

/// Стабильный top-k по logprob. При равных значениях выигрывает меньший token id.
pub fn top_k_candidates(logits: &[f32], k: usize) -> Vec<TreeDraftCandidate> {
    assert!(!logits.is_empty(), "logits must not be empty");
    assert!(k > 0, "branching must be positive");
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits
        .iter()
        .map(|&v| (v - max).exp())
        .sum::<f32>()
        .max(1e-30);
    let normalizer = exp_sum.ln() + max;

    let mut candidates: Vec<TreeDraftCandidate> = logits
        .iter()
        .enumerate()
        .map(|(token, &logit)| TreeDraftCandidate {
            token: token as u32,
            logprob: logit - normalizer,
        })
        .collect();
    candidates.sort_by(|a, b| {
        b.logprob
            .total_cmp(&a.logprob)
            .then_with(|| a.token.cmp(&b.token))
    });
    candidates.truncate(k);
    candidates
}

impl FlatTrie {
    /// Отбор узлов по бюджету.
    ///
    /// Узел с большим суммарным logprob получает приоритет. Возврат —
    /// отсортированные индексы, включая root, поэтому бюджет задаёт общее
    /// число узлов дерева, а не только драфт-токенов.
    pub fn prune_to_budget(&self, budget: usize) -> Vec<usize> {
        assert!(budget > 0, "budget must keep at least the root");

        let mut sums = vec![0f32; self.nodes.len()];
        let mut order: Vec<usize> = (1..self.nodes.len()).collect();
        // Индексы уже упорядочены по depth: parent всегда встречается раньше child.
        for &idx in &order {
            let parent = self.nodes[idx]
                .node
                .parent
                .expect("non-root nodes must have a parent");
            sums[idx] = sums[parent] + self.nodes[idx].node.logprob;
        }
        order.sort_by(|&a, &b| sums[b].total_cmp(&sums[a]));

        let mut kept = vec![false; self.nodes.len()];
        kept[0] = true;
        for &idx in order.iter().take(budget.saturating_sub(1)) {
            kept[idx] = true;
        }

        // Держать родителей всех выбранных узлов: дерево без разрывов —
        // обязательный инвариант для trie-маски и приёма пути.
        for idx in (1..self.nodes.len()).rev() {
            if kept[idx] {
                let mut parent = self.nodes[idx]
                    .node
                    .parent
                    .expect("non-root nodes must have a parent");
                while parent != 0 {
                    kept[parent] = true;
                    parent = self.nodes[parent]
                        .node
                        .parent
                        .expect("non-root nodes must have a parent");
                }
            }
        }

        (0..self.nodes.len()).filter(|&idx| kept[idx]).collect()
    }

    pub fn from_edges(edges: &[(usize, u32, f32)]) -> Self {
        let mut nodes = Vec::with_capacity(edges.len() + 1);
        nodes.push(FlatTrieNode {
            node: TrieNode {
                token: 0,
                parent: None,
                logprob: 0.0,
            },
            start: 0,
            end: 0,
            depth: 0,
        });

        for &(parent, token, logprob) in edges {
            let idx = nodes.len();
            nodes.push(FlatTrieNode {
                node: TrieNode {
                    token,
                    parent: Some(parent),
                    logprob,
                },
                start: idx,
                end: idx,
                depth: nodes[parent].depth + 1,
            });
        }

        // Пересчитать диапазоны поддеревьев после добавления всех узлов.
        let mut ranges = vec![(0usize, 0usize); nodes.len()];
        for i in (0..nodes.len()).rev() {
            let mut start = i;
            let mut end = i;
            for j in 0..nodes.len() {
                if nodes[j].node.parent == Some(i) {
                    start = start.min(ranges[j].0);
                    end = end.max(ranges[j].1);
                }
            }
            ranges[i] = (start, end);
        }
        for (i, range) in ranges.into_iter().enumerate() {
            nodes[i].start = range.0;
            nodes[i].end = range.1;
        }

        Self { nodes }
    }

    /// Построить дерево ветвящегося драфта.
    ///
    /// `next_candidates(parent)` возвращает кандидатов для продолжения узла.
    /// На каждом уровне выбираются лучшие кандидаты по суммарному logprob пути,
    /// поэтому общий префикс расходует бюджет один раз, а не дублируется в
    /// каждой ветке. `budget` включает root-sentinel.
    pub fn build_tree(
        spec: TreeDraftSpec,
        mut next_candidates: impl FnMut(usize) -> Vec<TreeDraftCandidate>,
    ) -> Self {
        assert!(spec.budget > 0, "budget must keep at least the root");
        assert!(spec.depth > 0, "depth must be positive");
        assert!(spec.branching > 0, "branching must be positive");

        let mut edges = Vec::new();
        let mut path_scores = vec![0f32];
        let mut frontier = vec![0usize];

        for _ in 0..spec.depth {
            if frontier.is_empty() || edges.len() + 1 >= spec.budget {
                break;
            }

            let mut proposals = Vec::new();
            for &parent in &frontier {
                for candidate in next_candidates(parent) {
                    proposals.push((parent, candidate));
                }
            }
            proposals.sort_by(|a, b| {
                let score_a = path_scores[a.0] + a.1.logprob;
                let score_b = path_scores[b.0] + b.1.logprob;
                score_b
                    .total_cmp(&score_a)
                    .then_with(|| a.1.token.cmp(&b.1.token))
            });

            let remaining = spec.budget - edges.len() - 1;
            let mut next_frontier = Vec::new();
            for (parent, candidate) in proposals.into_iter().take(remaining) {
                let index = edges.len() + 1;
                edges.push((parent, candidate.token, candidate.logprob));
                path_scores.push(path_scores[parent] + candidate.logprob);
                next_frontier.push(index);
            }
            frontier = next_frontier;
        }

        Self::from_edges(&edges)
    }

    /// Стейджинг для CUDA: токены, родители, глубины и диапазоны поддеревьев.
    pub fn staging(&self) -> TreeStaging {
        TreeStaging {
            tokens: self.nodes.iter().map(|n| n.node.token).collect(),
            parents: self
                .nodes
                .iter()
                .map(|n| n.node.parent.unwrap_or(0))
                .collect(),
            depths: self.nodes.iter().map(|n| n.depth as u32).collect(),
            starts: self.nodes.iter().map(|n| n.start as u32).collect(),
            ends: self.nodes.iter().map(|n| n.end as u32).collect(),
        }
    }

    pub fn nodes(&self) -> &[FlatTrieNode] {
        &self.nodes
    }

    /// Индексы прямых детей узла.
    pub fn children(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(move |(i, n)| i != &node && n.node.parent == Some(node))
            .map(|(i, _)| i)
    }

    /// Принять путь по фактическим токенам target.
    ///
    /// Root — sentinel-узел, поэтому первый target-токен сравнивается с детьми root.
    /// Возвращает индексы принятых узлов; пустой вектор — ни один draft-токен не принят.
    pub fn accept(&self, target: &[u32]) -> Vec<usize> {
        let mut current = 0;
        let mut accepted = Vec::new();
        for &token in target {
            let Some(next) = self
                .children(current)
                .find(|&i| self.nodes[i].node.token == token)
            else {
                break;
            };
            accepted.push(next);
            current = next;
        }
        accepted
    }

    /// Индексы поддерева узла (включая сам узел).
    ///
    /// Диапазон `[start, end]` — это линейная часть дерева, видимая узлом
    /// при trie-маске внимания: строка запроса node может читать свой prefix
    /// и все узлы этого диапазона, но не чужие ветки.
    pub fn subtree(&self, node: usize) -> std::ops::RangeInclusive<usize> {
        self.nodes[node].start..=self.nodes[node].end
    }

    /// Путь root → node (включая оба конца).
    ///
    /// Нужен для отката DeltaNet/KV к границе принятого префикса: accepted-путь
    /// удобно проверять от корня, а не от листа.
    pub fn path_to_root(&self, node: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(self.nodes[node].depth + 1);
        let mut current = Some(node);
        while let Some(idx) = current {
            path.push(idx);
            current = self.nodes[idx].node.parent;
        }
        path.reverse();
        path
    }

    /// Токены по индексам пути.
    pub fn path_tokens(&self, indices: &[usize]) -> Vec<u32> {
        indices.iter().map(|&i| self.nodes[i].node.token).collect()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_tree_ranges() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (2, 3, -3.0)]);
        assert_eq!(trie.len(), 4);
        let nodes = trie.nodes();
        assert_eq!(nodes[0].start, 0);
        assert_eq!(nodes[0].end, 3);
        assert_eq!(nodes[1].start, 1);
        assert_eq!(nodes[1].end, 3);
        assert_eq!(nodes[2].start, 2);
        assert_eq!(nodes[2].end, 3);
        assert_eq!(nodes[3].start, 3);
        assert_eq!(nodes[3].end, 3);
    }

    #[test]
    fn accept_follows_branch() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (1, 3, -3.0), (2, 4, -4.0)]);
        let accepted = trie.accept(&[1, 2, 4]);
        assert_eq!(accepted, vec![1, 2, 4]);
        assert_eq!(trie.path_tokens(&accepted), vec![1, 2, 4]);
    }

    #[test]
    fn accept_stops_on_mismatch() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (1, 3, -3.0)]);
        assert_eq!(trie.accept(&[1, 9]), vec![1]);
        assert_eq!(trie.accept(&[9, 1]), Vec::<usize>::new());
    }

    #[test]
    fn prune_to_budget_keeps_best_connected_path() {
        let trie = FlatTrie::from_edges(&[
            (0, 1, -1.0),
            (1, 2, -0.1),
            (1, 3, -4.0),
            (2, 4, -0.2),
            (0, 5, -0.5),
        ]);
        // Root + три лучших узла. Узел 3 отрезается как худшая ветка.
        let kept = trie.prune_to_budget(4);
        assert_eq!(kept, vec![0, 1, 2, 5]);
    }

    #[test]
    fn prune_to_budget_includes_parent_chain() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -1.0), (2, 3, -0.1), (0, 4, -3.0)]);
        // По бюджету выбирается узел 3, но его родительская цепь обязательна.
        let kept = trie.prune_to_budget(4);
        assert_eq!(kept, vec![0, 1, 2, 3]);
    }

    #[test]
    fn subtree_range_is_contiguous() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (1, 3, -3.0), (0, 4, -4.0)]);
        assert_eq!(trie.subtree(0), 0..=4);
        assert_eq!(trie.subtree(1), 1..=3);
        assert_eq!(trie.subtree(2), 2..=2);
        assert_eq!(trie.subtree(4), 4..=4);
    }

    #[test]
    fn path_to_root_follows_parents() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (2, 3, -3.0)]);
        assert_eq!(trie.path_to_root(3), vec![0, 1, 2, 3]);
    }

    #[test]
    fn top_k_candidates_prefers_probability_and_lower_id() {
        let candidates = top_k_candidates(&[1.0, 3.0, 3.0, 0.0], 3);
        assert_eq!(
            candidates
                .iter()
                .map(|candidate| candidate.token)
                .collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
        assert!(candidates[0].logprob >= candidates[1].logprob);
    }

    #[test]
    fn build_tree_respects_budget_and_depth() {
        let trie = FlatTrie::build_tree(
            TreeDraftSpec {
                budget: 5,
                depth: 4,
                branching: 2,
            },
            |parent| {
                top_k_candidates(
                    match parent {
                        0 => &[0.0, 2.0, 1.0],
                        1 => &[3.0, 1.0],
                        2 => &[2.0, 0.0],
                        _ => &[0.0, 1.0],
                    },
                    2,
                )
            },
        );
        assert_eq!(trie.len(), 5);
        assert_eq!(trie.nodes().last().unwrap().depth, 2);
    }

    #[test]
    fn staging_matches_flat_order() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (0, 3, -3.0)]);
        let staging = trie.staging();
        assert_eq!(staging.tokens, vec![0, 1, 2, 3]);
        assert_eq!(staging.parents, vec![0, 0, 1, 0]);
        assert_eq!(staging.depths, vec![0, 1, 2, 1]);
        assert_eq!(staging.starts, vec![0, 1, 2, 3]);
        assert_eq!(staging.ends, vec![3, 2, 2, 3]);
    }

    #[test]
    fn host_mask_allows_prefix_and_subtree_only() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (1, 2, -2.0), (0, 3, -3.0), (3, 4, -4.0)]);
        let staging = trie.staging();
        let prefix_len = 7;
        // Root-ключ (suffix 0) виден всем строкам дерева.
        assert!(trie_mask_allows(&staging, prefix_len, 1, 7));
        assert!(trie_mask_allows(&staging, prefix_len, 3, 7));
        // Ключ узла 1 виден строкам его поддерева 1..2, но не соседней ветке.
        assert!(trie_mask_allows(&staging, prefix_len, 1, 8));
        assert!(trie_mask_allows(&staging, prefix_len, 2, 8));
        assert!(!trie_mask_allows(&staging, prefix_len, 3, 8));
        // Ключ узла 3 виден только своей ветке 3..4.
        assert!(trie_mask_allows(&staging, prefix_len, 3, 10));
        assert!(trie_mask_allows(&staging, prefix_len, 4, 10));
        assert!(!trie_mask_allows(&staging, prefix_len, 1, 10));
    }

    #[test]
    fn branching_tree_ranges() {
        let trie = FlatTrie::from_edges(&[(0, 1, -1.0), (0, 2, -2.0), (1, 3, -3.0), (2, 4, -4.0)]);
        assert_eq!(trie.len(), 5);
        let nodes = trie.nodes();
        assert_eq!(nodes[0].start, 0);
        assert_eq!(nodes[0].end, 4);
        assert_eq!(nodes[1].start, 1);
        assert_eq!(nodes[1].end, 3);
        assert_eq!(nodes[2].start, 2);
        assert_eq!(nodes[2].end, 4);
        assert_eq!(nodes[3].start, 3);
        assert_eq!(nodes[3].end, 3);
        assert_eq!(nodes[4].start, 4);
        assert_eq!(nodes[4].end, 4);
    }
}
