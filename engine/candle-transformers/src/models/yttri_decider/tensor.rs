//! Тензорный путь для CUDA: тот же энкодер на тензорных операциях candle, f32. Устройство передаёт вызывающий —
//! этот модуль CUDA сам не поднимает (свежая среда CUDA стоит +365 МБ RSS и до 10 с на пустом JIT-кэше, на уже
//! активной — +75 МБ RSS и +40 МБ VRAM, готовность около 50 мс).
//!
//! Строка эмбеддингов (таблица Q8_0 из отображения, позиции HF, LayerNorm) готовится на хосте и уходит на устройство
//! одним `L × hidden` буфером; дальше всё на GPU, обратно приходят только логиты. CLS-only в последнем слое и
//! `retain_default_mempool` здесь не используются: на GPU они выигрыша не дают (замер S4: 3,2–3,3 мс при любом варианте).
//!
//! Скорость: подряд (L = 256) ≈ 3,3 мс; после простоя дольше 7–8 с видеокарта уходит в P8 (210–330 МГц), и тот же
//! проход занимает ≈ 31 мс — медленнее lean-CPU, но в норме p95 ≤ 40 мс. Это свойство железа, оно документируется и не лечится.

use super::gguf::{DeciderConfig, Gguf, Table};
use candle::{bail, Device, Result, Tensor};
use candle_nn::ops::{layer_norm, softmax_last_dim};
use std::sync::{Arc, Mutex};

/// Линейный слой с весом, сохранённым как `[in, out]` (транспонированный вид: cuBLAS читает его без копии).
struct Lin {
    wt: Tensor,
    b: Tensor,
}

impl Lin {
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.wt)?.broadcast_add(&self.b)
    }
}

struct Layer {
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    i: Lin,
    out: Lin,
    ln1_g: Tensor,
    ln1_b: Tensor,
    ln2_g: Tensor,
    ln2_b: Tensor,
}

struct Head {
    dense: Lin,
    out: Lin,
}

pub(super) struct TensorPath {
    dev: Device,
    hid: usize,
    nh: usize,
    hd: usize,
    eps: f32,
    pad: u32,
    table: Table,
    pos_emb: Vec<f32>,
    tok0: Vec<f32>,
    emb_g: Tensor,
    emb_b: Tensor,
    layers: Vec<Layer>,
    heads: Vec<Head>,
    resident: usize,
    /// Вызовы одной модели идут по очереди: одно устройство и общий handle cuBLAS. Замок держится на весь `forward` без
    /// риска для пула rayon: на этом пути rayon не используется (ядра CUDA и копии), вложенных ожиданий нет.
    lock: Mutex<()>,
}

/// Матрица (Q8_0, деквантуется на хосте, или F32) + bias на устройство; `scale` домножает и то, и другое
/// (1/√head_dim для Q: так softmax получает уже масштабированные scores).
fn lin(g: &Gguf, dev: &Device, wn: &str, bn: &str, scale: f32, quantized: bool) -> Result<Lin> {
    let mut w = if quantized {
        g.q8_to_f32(wn)?
    } else {
        g.f32_vec(wn)?
    };
    let mut b = g.f32_vec(bn)?;
    if scale != 1.0 {
        for v in w.iter_mut().chain(b.iter_mut()) {
            *v *= scale;
        }
    }
    let (rows, cols) = (b.len(), w.len() / b.len());
    let w = Tensor::from_vec(w, (rows, cols), dev)?;
    Ok(Lin {
        wt: w.t()?,
        b: Tensor::from_vec(b, rows, dev)?,
    })
}

fn vec_t(g: &Gguf, dev: &Device, name: &str) -> Result<Tensor> {
    let v = g.f32_vec(name)?;
    let n = v.len();
    Tensor::from_vec(v, n, dev)
}

fn bytes(t: &Tensor) -> usize {
    t.elem_count() * 4
}

impl TensorPath {
    pub(super) fn load(g: &Arc<Gguf>, dev: &Device) -> Result<Self> {
        if !dev.is_cuda() {
            bail!(
                "yttri_decider: тензорный путь работает только на CUDA, получено {:?}",
                dev.location()
            );
        }
        let cfg: &DeciderConfig = &g.cfg;
        let scale = 1.0 / (cfg.head_dim() as f32).sqrt();
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = format!("encoder.layer.{i}");
            let wn = |s: &str| format!("{p}.{s}.weight");
            let bn = |s: &str| format!("{p}.{s}.bias");
            let l = |s: &str, scale: f32| lin(g, dev, &wn(s), &bn(s), scale, true);
            layers.push(Layer {
                q: l("attention.self.query", scale)?,
                k: l("attention.self.key", 1.0)?,
                v: l("attention.self.value", 1.0)?,
                o: l("attention.output.dense", 1.0)?,
                i: l("intermediate.dense", 1.0)?,
                out: l("output.dense", 1.0)?,
                ln1_g: vec_t(g, dev, &wn("attention.output.LayerNorm"))?,
                ln1_b: vec_t(g, dev, &bn("attention.output.LayerNorm"))?,
                ln2_g: vec_t(g, dev, &wn("output.LayerNorm"))?,
                ln2_b: vec_t(g, dev, &bn("output.LayerNorm"))?,
            });
        }
        let mut heads = Vec::with_capacity(cfg.heads.len());
        for (qid, _) in &cfg.heads {
            let h = |s: &str| {
                lin(
                    g,
                    dev,
                    &format!("heads.{qid}.{s}.weight"),
                    &format!("heads.{qid}.{s}.bias"),
                    1.0,
                    false,
                )
            };
            heads.push(Head {
                dense: h("dense")?,
                out: h("out_proj")?,
            });
        }
        let (emb_g, emb_b) = (
            vec_t(g, dev, "embeddings.LayerNorm.weight")?,
            vec_t(g, dev, "embeddings.LayerNorm.bias")?,
        );
        let lin_bytes = |l: &Lin| bytes(&l.wt) + bytes(&l.b);
        let resident = bytes(&emb_g)
            + bytes(&emb_b)
            + layers
                .iter()
                .map(|l| {
                    [&l.q, &l.k, &l.v, &l.o, &l.i, &l.out]
                        .iter()
                        .map(|x| lin_bytes(x))
                        .sum::<usize>()
                        + bytes(&l.ln1_g)
                        + bytes(&l.ln1_b)
                        + bytes(&l.ln2_g)
                        + bytes(&l.ln2_b)
                })
                .sum::<usize>()
            + heads
                .iter()
                .map(|h| lin_bytes(&h.dense) + lin_bytes(&h.out))
                .sum::<usize>();
        // загрузка асинхронна: возвращаемся, когда веса уже на устройстве
        dev.synchronize()?;
        Ok(Self {
            dev: dev.clone(),
            hid: cfg.hidden,
            nh: cfg.attention_heads,
            hd: cfg.head_dim(),
            eps: cfg.eps,
            pad: cfg.pad_id,
            table: g.table()?,
            pos_emb: g.f32_vec("embeddings.position_embeddings.weight")?,
            tok0: g.f32_vec("embeddings.token_type_embeddings.weight")?,
            emb_g,
            emb_b,
            layers,
            heads,
            resident,
            lock: Mutex::new(()),
        })
    }

    /// Байт весов на устройстве.
    pub(super) fn resident_bytes(&self) -> usize {
        self.resident
    }

    fn layer(&self, ly: &Layer, h: &Tensor, l: usize) -> Result<Tensor> {
        let (hid, nh, hd) = (self.hid, self.nh, self.hd);
        let (q, k, v) = (ly.q.fwd(h)?, ly.k.fwd(h)?, ly.v.fwd(h)?);
        // cuBLAS принимает только плотные операнды батча: перестановки голов материализуем
        let qh = q.reshape((l, nh, hd))?.transpose(0, 1)?.contiguous()?;
        let kt = k.reshape((l, nh, hd))?.permute((1, 2, 0))?.contiguous()?;
        let vh = v.reshape((l, nh, hd))?.transpose(0, 1)?.contiguous()?;
        let probs = softmax_last_dim(&qh.matmul(&kt)?)?; // Wq и bq уже домножены на 1/√head_dim
        let ctx = probs
            .matmul(&vh)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((l, hid))?;
        let a = layer_norm(&(ly.o.fwd(&ctx)? + h)?, &ly.ln1_g, &ly.ln1_b, self.eps)?;
        let f = ly.i.fwd(&a)?.gelu_erf()?;
        layer_norm(&(ly.out.fwd(&f)? + &a)?, &ly.ln2_g, &ly.ln2_b, self.eps)
    }

    /// Логиты всех голов подряд в `out`. Вход уже проверен (`YttriDecider::forward`).
    pub(super) fn forward(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        // в замке нет состояния, паника прежнего вызова ему не мешает
        let _turn = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        let (l, hid) = (ids.len(), self.hid);
        let mut x = vec![0f32; l * hid];
        let mut n = 0u32;
        for (row, &id) in x.chunks_exact_mut(hid).zip(ids) {
            // позиции HF: cumsum(ids != pad) · (ids != pad) + pad
            let pos = if id == self.pad {
                self.pad
            } else {
                n += 1;
                self.pad + n
            };
            let pos = pos as usize;
            self.table.row_into(id, row)?;
            let p = &self.pos_emb[pos * hid..(pos + 1) * hid];
            for j in 0..hid {
                row[j] = (row[j] + self.tok0[j]) + p[j];
            }
        }
        let x = Tensor::from_vec(x, (l, hid), &self.dev)?;
        let mut h = layer_norm(&x, &self.emb_g, &self.emb_b, self.eps)?;
        for ly in &self.layers {
            h = self.layer(ly, &h, l)?;
        }
        let cls = h.narrow(0, 0, 1)?;
        let mut zs = Vec::with_capacity(self.heads.len());
        for hd in &self.heads {
            zs.push(hd.out.fwd(&hd.dense.fwd(&cls)?.tanh()?)?);
        }
        let z = Tensor::cat(&zs, 1)?.flatten_all()?.to_vec1::<f32>()?;
        if z.len() != out.len() {
            bail!("yttri_decider: {} выходов вместо {}", z.len(), out.len());
        }
        out.copy_from_slice(&z);
        Ok(())
    }
}
