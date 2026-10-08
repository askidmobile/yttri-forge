//! Lean-путь CPU: энкодер decider без тензорного слоя candle. Сырые срезы f32, заранее выделенное рабочее
//! пространство (в установившемся режиме аллокаций нет, кроме внутренних буферов `gemm`), `gemm` напрямую, слитые
//! ядра (bias+GELU, bias+residual+LayerNorm), внимание блоками (голова × 64 строки запроса) без буфера scores на всю
//! длину, QKV одним gemm, в последнем слое только строка CLS (B = 1, маски нет — результат тот же).
//! Веса — f32, деквантованы из Q8_0 при загрузке (`QMatMul` отклонён: в 7,6 раза медленнее и Δz 4,2e-2 при допуске 1e-3).
//!
//! Потоки — от вызывающего: все параллельные операции (rayon, `gemm` с `Parallelism::Rayon(0)`) идут в текущем пуле,
//! то есть в том, внутри которого вызвали `forward` (`ThreadPool::install`); вне пула — в глобальном.

use super::gguf::{DeciderConfig, Gguf, Table};
use super::kern::{self, Isa};
use candle::Result;
use gemm::Parallelism;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// Строк запроса на одну задачу внимания.
const RB: usize = 64;
/// Строк на одну задачу поэлементных ядер.
const RC: usize = 32;

struct Layer {
    wqkv: Vec<f32>, // [3·hid, hid]: Wq (домножена на 1/√head_dim), Wk, Wv
    bqkv: Vec<f32>, // [3·hid]
    wo: Vec<f32>,
    bo: Vec<f32>,
    ln1_g: Vec<f32>,
    ln1_b: Vec<f32>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    ln2_g: Vec<f32>,
    ln2_b: Vec<f32>,
}

struct Head {
    width: usize,
    out_w: Vec<f32>, // [width, hid]
    out_b: Vec<f32>,
}

/// Рабочее пространство: растёт только если вход длиннее всех прежних.
struct Ws {
    cap: usize,
    x: Vec<f32>,
    qkv: Vec<f32>,
    ctx: Vec<f32>,
    tmp: Vec<f32>,
    ffn: Vec<f32>,
    sc: Vec<f32>,
    pos: Vec<u32>,
    z: Vec<f32>,
}

impl Ws {
    fn new() -> Self {
        Ws {
            cap: 0,
            x: vec![],
            qkv: vec![],
            ctx: vec![],
            tmp: vec![],
            ffn: vec![],
            sc: vec![],
            pos: vec![],
            z: vec![],
        }
    }

    fn ensure(&mut self, m: &Lean, rows: usize) {
        if rows <= self.cap {
            return;
        }
        self.cap = rows;
        self.x.resize(rows * m.hid, 0.0);
        self.qkv.resize(rows * 3 * m.hid, 0.0);
        self.ctx.resize(rows * m.hid, 0.0);
        self.tmp.resize(rows * m.hid, 0.0);
        self.ffn.resize(rows * m.ff, 0.0);
        self.sc.resize(m.nh * rows, 0.0);
        self.pos.resize(rows, 0);
        self.z.resize(m.heads.len() * m.hid, 0.0);
    }
}

pub(super) struct Lean {
    isa: Isa,
    hid: usize,
    ff: usize,
    nh: usize,
    hd: usize,
    eps: f32,
    pad: u32,
    layers: Vec<Layer>,
    pos_emb: Vec<f32>,
    tok0: Vec<f32>,
    emb_g: Vec<f32>,
    emb_b: Vec<f32>,
    dense_w: Vec<f32>, // dense всех голов подряд: [голов·hid, hid]
    dense_b: Vec<f32>,
    heads: Vec<Head>,
    table: Table,
    ws: Mutex<Ws>,
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: задачи внимания пишут в непересекающиеся блоки (голова × диапазон строк) одного буфера
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    // метод, а не поле: замыкание должно захватить всю обёртку (Sync), а не голый *mut f32
    #[inline]
    fn ptr(&self) -> *mut f32 {
        self.0
    }
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// C[m×n, ld=ldc] = A[m×k, ld=lda] · Wᵀ, W — [n×k] построчно (раскладка HF), перезаписывает C.
#[allow(clippy::too_many_arguments)]
fn mm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    lda: usize,
    w: &[f32],
    c: &mut [f32],
    ldc: usize,
    par: Parallelism,
) {
    assert!(
        m >= 1 && a.len() >= (m - 1) * lda + k && w.len() >= n * k && c.len() >= (m - 1) * ldc + n
    );
    // SAFETY: границы a, w, c проверены выше; read_dst = false — C не читается
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            c.as_mut_ptr(),
            1,
            ldc as isize,
            false,
            a.as_ptr(),
            1,
            lda as isize,
            w.as_ptr(),
            k as isize,
            1,
            0.0f32,
            1.0f32,
            false,
            false,
            false,
            par,
        )
    }
}

const POOL: Parallelism = Parallelism::Rayon(0); // 0 — число потоков текущего пула rayon

impl Lean {
    pub(super) fn load(g: &Arc<Gguf>, isa: Isa) -> Result<Self> {
        let cfg: &DeciderConfig = &g.cfg;
        let hid = cfg.hidden;
        let hd = cfg.head_dim();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layer.{i}");
            let w = |s: &str| format!("{p}.{s}.weight");
            let b = |s: &str| format!("{p}.{s}.bias");
            // 1/√head_dim вшит в Wq и bq: scores получаются уже масштабированными
            let mut wq = g.q8_to_f32(&w("attention.self.query"))?;
            let mut bq = g.f32_vec(&b("attention.self.query"))?;
            for v in wq.iter_mut().chain(bq.iter_mut()) {
                *v *= scale;
            }
            let mut wqkv = Vec::with_capacity(3 * hid * hid);
            wqkv.extend_from_slice(&wq);
            wqkv.extend_from_slice(&g.q8_to_f32(&w("attention.self.key"))?);
            wqkv.extend_from_slice(&g.q8_to_f32(&w("attention.self.value"))?);
            let mut bqkv = Vec::with_capacity(3 * hid);
            bqkv.extend_from_slice(&bq);
            bqkv.extend_from_slice(&g.f32_vec(&b("attention.self.key"))?);
            bqkv.extend_from_slice(&g.f32_vec(&b("attention.self.value"))?);
            layers.push(Layer {
                wqkv,
                bqkv,
                wo: g.q8_to_f32(&w("attention.output.dense"))?,
                bo: g.f32_vec(&b("attention.output.dense"))?,
                ln1_g: g.f32_vec(&w("attention.output.LayerNorm"))?,
                ln1_b: g.f32_vec(&b("attention.output.LayerNorm"))?,
                w1: g.q8_to_f32(&w("intermediate.dense"))?,
                b1: g.f32_vec(&b("intermediate.dense"))?,
                w2: g.q8_to_f32(&w("output.dense"))?,
                b2: g.f32_vec(&b("output.dense"))?,
                ln2_g: g.f32_vec(&w("output.LayerNorm"))?,
                ln2_b: g.f32_vec(&b("output.LayerNorm"))?,
            });
        }
        let (mut dense_w, mut dense_b, mut heads) = (Vec::new(), Vec::new(), Vec::new());
        for (qid, width) in &cfg.heads {
            dense_w.extend_from_slice(&g.f32_vec(&format!("heads.{qid}.dense.weight"))?);
            dense_b.extend_from_slice(&g.f32_vec(&format!("heads.{qid}.dense.bias"))?);
            heads.push(Head {
                width: *width,
                out_w: g.f32_vec(&format!("heads.{qid}.out_proj.weight"))?,
                out_b: g.f32_vec(&format!("heads.{qid}.out_proj.bias"))?,
            });
        }
        let mut m = Lean {
            isa,
            hid,
            ff: cfg.intermediate,
            nh: cfg.attention_heads,
            hd,
            eps: cfg.eps,
            pad: cfg.pad_id,
            layers,
            pos_emb: g.f32_vec("embeddings.position_embeddings.weight")?,
            tok0: g.f32_vec("embeddings.token_type_embeddings.weight")?,
            emb_g: g.f32_vec("embeddings.LayerNorm.weight")?,
            emb_b: g.f32_vec("embeddings.LayerNorm.bias")?,
            dense_w,
            dense_b,
            heads,
            table: g.table()?,
            ws: Mutex::new(Ws::new()),
        };
        // штатный вход (до max_tokens) считается без аллокаций; более длинный — растит рабочее пространство
        let mut ws = Ws::new();
        ws.ensure(&m, cfg.max_tokens);
        m.ws = Mutex::new(ws);
        Ok(m)
    }

    /// Байт в рабочих весах и буферах (без отображённой таблицы эмбеддингов).
    pub(super) fn resident_bytes(&self) -> usize {
        let mut n = self.pos_emb.len()
            + self.tok0.len()
            + self.emb_g.len()
            + self.emb_b.len()
            + self.dense_w.len()
            + self.dense_b.len();
        n += self
            .heads
            .iter()
            .map(|h| h.out_w.len() + h.out_b.len())
            .sum::<usize>();
        for l in &self.layers {
            n += l.wqkv.len()
                + l.bqkv.len()
                + l.wo.len()
                + l.bo.len()
                + l.ln1_g.len()
                + l.ln1_b.len();
            n += l.w1.len() + l.b1.len() + l.w2.len() + l.b2.len() + l.ln2_g.len() + l.ln2_b.len();
        }
        let ws = self.ws.lock().unwrap_or_else(|e| e.into_inner());
        n += ws.x.len()
            + ws.qkv.len()
            + ws.ctx.len()
            + ws.tmp.len()
            + ws.ffn.len()
            + ws.sc.len()
            + ws.z.len()
            + ws.pos.len();
        n * 4
    }

    /// Логиты всех голов подряд в `out`. Вход уже проверен (`YttriDecider::forward`): 1 ≤ L, id < vocab, pad_id + L < positions.
    pub(super) fn forward(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        let (l, hid, isa, eps) = (ids.len(), self.hid, self.isa, self.eps);
        // буферы целиком перезаписываются на каждом проходе: после паники прежнего вызова мьютекс можно брать как есть
        let mut guard = self.ws.lock().unwrap_or_else(|e| e.into_inner());
        let ws = &mut *guard;
        ws.ensure(self, l);

        // позиции HF: cumsum(ids != pad) · (ids != pad) + pad; литерал <pad> занимает позицию pad и не сдвигает счёт
        let mut n = 0u32;
        for (p, &id) in ws.pos[..l].iter_mut().zip(ids) {
            *p = if id == self.pad {
                self.pad
            } else {
                n += 1;
                self.pad + n
            };
        }

        {
            let Ws { x, pos, .. } = &mut *ws;
            let pos = &pos[..l];
            x[..l * hid].par_chunks_mut(hid).enumerate().try_for_each(
                |(i, row)| -> Result<()> {
                    self.table.row_into(ids[i], row)?;
                    let p = &self.pos_emb[pos[i] as usize * hid..(pos[i] as usize + 1) * hid];
                    for j in 0..hid {
                        row[j] = (row[j] + self.tok0[j]) + p[j];
                    }
                    Ok(())
                },
            )?;
            x[..l * hid].par_chunks_mut(RC * hid).for_each(|c| {
                kern::layer_norm_rows(isa, c, None, None, &self.emb_g, &self.emb_b, hid, eps)
            });
        }

        for (li, ly) in self.layers.iter().enumerate() {
            if li + 1 == self.layers.len() {
                self.layer_cls(ly, l, ws);
            } else {
                self.layer_full(ly, l, ws);
            }
        }

        // головы: z = out_proj(tanh(dense(h[0]))), h[0] — позиция CLS
        let Ws { x, z, .. } = &mut *ws;
        let nq = self.heads.len();
        let z = &mut z[..nq * hid];
        mm(
            1,
            nq * hid,
            hid,
            &x[..hid],
            hid,
            &self.dense_w,
            z,
            nq * hid,
            POOL,
        );
        for (v, &b) in z.iter_mut().zip(self.dense_b.iter()) {
            *v = (*v + b).tanh();
        }
        let mut off = 0;
        for (h, hd) in self.heads.iter().enumerate() {
            let o = &mut out[off..off + hd.width];
            mm(
                1,
                hd.width,
                hid,
                &z[h * hid..(h + 1) * hid],
                hid,
                &hd.out_w,
                o,
                hd.width,
                Parallelism::None,
            );
            for (v, &b) in o.iter_mut().zip(hd.out_b.iter()) {
                *v += b;
            }
            off += hd.width;
        }
        Ok(())
    }

    fn layer_full(&self, ly: &Layer, l: usize, ws: &mut Ws) {
        let (hid, ff, isa, eps) = (self.hid, self.ff, self.isa, self.eps);
        let ld = 3 * hid;
        let Ws {
            x,
            qkv,
            ctx,
            tmp,
            ffn,
            ..
        } = ws;
        mm(
            l,
            ld,
            hid,
            &x[..l * hid],
            hid,
            &ly.wqkv,
            &mut qkv[..l * ld],
            ld,
            POOL,
        );
        qkv[..l * ld]
            .par_chunks_mut(RC * ld)
            .for_each(|c| kern::add_bias(isa, c, &ly.bqkv, ld));
        self.attention(&qkv[..l * ld], &mut ctx[..l * hid], l);
        mm(
            l,
            hid,
            hid,
            &ctx[..l * hid],
            hid,
            &ly.wo,
            &mut tmp[..l * hid],
            hid,
            POOL,
        );
        tmp[..l * hid]
            .par_chunks_mut(RC * hid)
            .zip(x[..l * hid].par_chunks(RC * hid))
            .for_each(|(c, r)| {
                kern::layer_norm_rows(
                    isa,
                    c,
                    Some(r),
                    Some(&ly.bo),
                    &ly.ln1_g,
                    &ly.ln1_b,
                    hid,
                    eps,
                )
            });
        mm(
            l,
            ff,
            hid,
            &tmp[..l * hid],
            hid,
            &ly.w1,
            &mut ffn[..l * ff],
            ff,
            POOL,
        );
        ffn[..l * ff]
            .par_chunks_mut(RC * ff)
            .for_each(|c| kern::gelu_bias(isa, c, &ly.b1, ff));
        mm(
            l,
            hid,
            ff,
            &ffn[..l * ff],
            ff,
            &ly.w2,
            &mut x[..l * hid],
            hid,
            POOL,
        );
        x[..l * hid]
            .par_chunks_mut(RC * hid)
            .zip(tmp[..l * hid].par_chunks(RC * hid))
            .for_each(|(c, r)| {
                kern::layer_norm_rows(
                    isa,
                    c,
                    Some(r),
                    Some(&ly.b2),
                    &ly.ln2_g,
                    &ly.ln2_b,
                    hid,
                    eps,
                )
            });
    }

    /// Последний слой только для строки 0 (CLS): K и V по всем строкам, остальное — одна строка.
    fn layer_cls(&self, ly: &Layer, l: usize, ws: &mut Ws) {
        let (hid, ff, nh, hd, isa, eps) = (self.hid, self.ff, self.nh, self.hd, self.isa, self.eps);
        let kv = 2 * hid;
        let Ws {
            x,
            qkv,
            ctx,
            tmp,
            ffn,
            sc,
            ..
        } = ws;
        mm(
            l,
            kv,
            hid,
            &x[..l * hid],
            hid,
            &ly.wqkv[hid * hid..],
            &mut qkv[..l * kv],
            kv,
            POOL,
        );
        qkv[..l * kv]
            .par_chunks_mut(RC * kv)
            .for_each(|c| kern::add_bias(isa, c, &ly.bqkv[hid..], kv));
        // q0 → tmp[..hid] (Wq и bq уже домножены на 1/√head_dim)
        mm(
            1,
            hid,
            hid,
            &x[..hid],
            hid,
            &ly.wqkv[..hid * hid],
            &mut tmp[..hid],
            hid,
            Parallelism::None,
        );
        for (v, &b) in tmp[..hid].iter_mut().zip(ly.bqkv[..hid].iter()) {
            *v += b;
        }
        let sc = &mut sc[..nh * l];
        for h in 0..nh {
            let q = &tmp[h * hd..(h + 1) * hd];
            for j in 0..l {
                let k = &qkv[j * kv + h * hd..j * kv + (h + 1) * hd];
                sc[h * l + j] = q.iter().zip(k).map(|(a, b)| a * b).sum();
            }
        }
        kern::softmax_rows(isa, sc, l);
        for h in 0..nh {
            let c = &mut ctx[h * hd..(h + 1) * hd];
            c.fill(0.0);
            for j in 0..l {
                let p = sc[h * l + j];
                let v = &qkv[j * kv + hid + h * hd..j * kv + hid + (h + 1) * hd];
                for (cv, &vv) in c.iter_mut().zip(v) {
                    *cv += p * vv;
                }
            }
        }
        mm(
            1,
            hid,
            hid,
            &ctx[..hid],
            hid,
            &ly.wo,
            &mut tmp[..hid],
            hid,
            Parallelism::None,
        );
        kern::layer_norm_rows(
            isa,
            &mut tmp[..hid],
            Some(&x[..hid]),
            Some(&ly.bo),
            &ly.ln1_g,
            &ly.ln1_b,
            hid,
            eps,
        );
        mm(
            1,
            ff,
            hid,
            &tmp[..hid],
            hid,
            &ly.w1,
            &mut ffn[..ff],
            ff,
            POOL,
        );
        kern::gelu_bias(isa, &mut ffn[..ff], &ly.b1, ff);
        mm(1, hid, ff, &ffn[..ff], ff, &ly.w2, &mut x[..hid], hid, POOL);
        kern::layer_norm_rows(
            isa,
            &mut x[..hid],
            Some(&tmp[..hid]),
            Some(&ly.b2),
            &ly.ln2_g,
            &ly.ln2_b,
            hid,
            eps,
        );
    }

    /// Внимание: задачи (голова, блок из RB строк). Scores блока живут в TLS-буфере (RB·L·4 байт, в L2), в общую память
    /// пишется только контекст. Q уже домножен на 1/√head_dim; маски нет (B = 1, литерал `<pad>` участвует как токен).
    fn attention(&self, qkv: &[f32], ctx: &mut [f32], l: usize) {
        let (hid, nh, hd, isa) = (self.hid, self.nh, self.hd, self.isa);
        let ld = 3 * hid;
        let nrb = l.div_ceil(RB);
        let cp = SendPtr(ctx.as_mut_ptr());
        assert!(qkv.len() >= l * ld && ctx.len() >= l * hid);
        (0..nh * nrb).into_par_iter().with_min_len(1).for_each(|t| {
            let (h, rb) = (t / nrb, t % nrb);
            let r0 = rb * RB;
            let rows = RB.min(l - r0);
            SCRATCH.with(|s| {
                let mut s = s.borrow_mut();
                if s.len() < RB * l {
                    s.resize(RB * l, 0.0);
                }
                let sc = &mut s[..rows * l];
                let q = &qkv[r0 * ld + h * hd..];
                let k = &qkv[hid + h * hd..];
                let v = &qkv[2 * hid + h * hd..];
                // SAFETY: чтение в границах qkv (последний элемент (r0+rows-1)·ld + (h+1)·hd + 2·hid ≤ l·ld),
                // запись — в sc и в блок ctx [r0..r0+rows) × [h·hd..(h+1)·hd), который принадлежит только этой задаче
                unsafe {
                    // scores[rows×l] = Q_h[rows×hd] · K_hᵀ
                    gemm::gemm(
                        rows,
                        l,
                        hd,
                        sc.as_mut_ptr(),
                        1,
                        l as isize,
                        false,
                        q.as_ptr(),
                        1,
                        ld as isize,
                        k.as_ptr(),
                        ld as isize,
                        1,
                        0.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
                kern::softmax_rows(isa, sc, l);
                unsafe {
                    // ctx[rows×hd] = P[rows×l] · V_h[l×hd], запись напрямую в [l×hid]
                    gemm::gemm(
                        rows,
                        hd,
                        l,
                        cp.ptr().add(r0 * hid + h * hd),
                        1,
                        hid as isize,
                        false,
                        sc.as_ptr(),
                        1,
                        l as isize,
                        v.as_ptr(),
                        1,
                        ld as isize,
                        0.0f32,
                        1.0f32,
                        false,
                        false,
                        false,
                        Parallelism::None,
                    );
                }
            });
        });
    }
}
