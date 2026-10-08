//! Фикстура decider-tiny для модуля `yttri_decider` (T-750, Ф3c, поток C2; `docs/integration-yttri.md` §10.9 в yttri-decider):
//! hidden 64, две головы внимания по 32, два слоя, intermediate 128, словарь 256 первых id e5, позиции 514, pad_id 1,
//! **eps LayerNorm 1e-2** (боевой 1e-5, крупный нарочно: golden ловит чужой eps), четыре головы (9, 1, 1, 53).
//!
//! Тест не игнорируется: в forge нет CI, а модуль исполняется в процессе Yttri (П14). Копия фикстуры — байт в байт из
//! `yttri-decider/bench/fixtures/decider-tiny`; оба sha256 закреплены литералами ниже и в §10.9 (файл и хеш внутри
//! `golden.json` дрейфуют вместе, поэтому хеш из самого golden пару не проверяет).
//!
//! Проверяется: хеши, golden на 14 входах (допуск `max_abs_dz` 2e-4 из файла) на обоих наборах ядер, отказы на четырёх
//! `reject` и на соседних граничных входах, повторяемость при разных пулах потоков и при параллельных вызовах,
//! строгость читалки на испорченных копиях файла и **чувствительность самого golden** к eps / GELU / позициям /
//! маске по `<pad>` — на независимом эталоне (naive f64, читает те же байты файла).

use candle::quantized::gguf_file::Content;
use candle::{Device, Result};
use candle_transformers::models::yttri_decider::YttriDecider;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MODEL_SHA256: &str = "71c184ccc00ff07bb57319612cb3da9ad55196d29c37bb989f5dce1328a601e2";
const GOLDEN_SHA256: &str = "a5f50db4e5439023ead2a9b494ed4ee1a537bc1f5cbe4c63a226ab54052e4e15";

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/decider-tiny")
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn golden() -> Value {
    serde_json::from_slice(&std::fs::read(dir().join("golden.json")).unwrap()).unwrap()
}

fn load_cpu() -> YttriDecider {
    YttriDecider::load(dir().join("model.gguf"), &Device::Cpu).unwrap()
}

fn ids_of(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

/// Логиты случая golden по головам в порядке `outputs`.
fn expected(case: &Value, outputs: &[(String, usize)]) -> Vec<Vec<f64>> {
    outputs
        .iter()
        .map(|(name, n)| {
            let z: Vec<f64> = case["logits"][name]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            assert_eq!(z.len(), *n, "{name}");
            z
        })
        .collect()
}

fn max_dz(got: &[Vec<f32>], want: &[Vec<f64>]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .flat_map(|(g, w)| g.iter().zip(w).map(|(a, b)| (*a as f64 - b).abs()))
        .fold(0.0, f64::max)
}

/// Прогон всех случаев golden: наибольшее отклонение по каждому случаю.
fn run_golden(model: &YttriDecider, g: &Value) -> Vec<(String, f64)> {
    g["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let got = model.forward(&ids_of(&c["ids"])).unwrap();
            (
                c["name"].as_str().unwrap().to_string(),
                max_dz(&got, &expected(c, model.outputs())),
            )
        })
        .collect()
}

fn tolerance(g: &Value) -> f64 {
    g["tolerance"]["max_abs_dz"].as_f64().unwrap()
}

// ------------------------------------------------------------------------------------------------ файлы и контракт

#[test]
fn fixture_files_match_the_pinned_hashes() {
    let model = std::fs::read(dir().join("model.gguf")).unwrap();
    let golden_bytes = std::fs::read(dir().join("golden.json")).unwrap();
    assert_eq!(sha256_hex(&model), MODEL_SHA256);
    assert_eq!(sha256_hex(&golden_bytes), GOLDEN_SHA256);
    let g = golden();
    assert_eq!(g["model"]["sha256"].as_str().unwrap(), MODEL_SHA256);
    assert_eq!(g["format"], 1);
    assert!(
        !golden_bytes.contains(&b'\r'),
        "golden.json должен быть с LF (.gitattributes)"
    );
    assert!(golden_bytes.ends_with(b"\n"));
}

#[test]
fn config_and_outputs_follow_the_file() {
    let m = load_cpu();
    let c = m.config();
    assert_eq!(
        (
            c.hidden,
            c.layers,
            c.attention_heads,
            c.intermediate,
            c.head_dim()
        ),
        (64, 2, 2, 128, 32)
    );
    assert_eq!(
        (c.positions, c.vocab, c.pad_id, c.max_tokens),
        (514, 256, 1, 256)
    );
    assert!(
        (c.eps - 1e-2).abs() < 1e-9,
        "eps из KV, а не константа: {}",
        c.eps
    );
    assert_eq!(
        (
            c.model_version.as_str(),
            c.quant.as_str(),
            c.registry_rev,
            c.state_version
        ),
        ("0.0.1-tiny", "q8_0", 1, 1)
    );
    let heads: Vec<_> = c.heads.iter().map(|(q, n)| (q.as_str(), *n)).collect();
    assert_eq!(
        heads,
        [
            ("mail_category", 9),
            ("mail_needs_attention", 1),
            ("payment_problem", 1),
            ("finance_category", 53)
        ]
    );
    // порядок и имена выходов — как в golden.outputs
    let want: Vec<(String, usize)> = golden()["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            (
                o[0].as_str().unwrap().to_string(),
                o[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    assert_eq!(m.outputs(), want.as_slice());
    assert!(m.resident_bytes() > 0);
}

// ------------------------------------------------------------------------------------------------ golden

#[test]
fn golden_cases_match_the_reference() {
    let (m, g) = (load_cpu(), golden());
    let tol = tolerance(&g);
    let rows = run_golden(&m, &g);
    assert_eq!(rows.len(), 14);
    let worst = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    eprintln!("lean-CPU (best isa): max |dz| = {worst:.3e} (допуск {tol:e}); по случаям: {rows:?}");
    for (name, dz) in &rows {
        assert!(*dz <= tol, "{name}: max |dz| = {dz:e} > {tol:e}");
    }
}

#[test]
fn golden_cases_match_with_baseline_kernels() {
    // запасной набор ядер (SSE2 на x86-64) — то, что считает на CPU без AVX2
    let (g, tol) = (golden(), tolerance(&golden()));
    let m =
        YttriDecider::load_with_baseline_kernels(dir().join("model.gguf"), &Device::Cpu).unwrap();
    let rows = run_golden(&m, &g);
    let worst = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    eprintln!("lean-CPU (baseline kernels): max |dz| = {worst:.3e}");
    for (name, dz) in &rows {
        assert!(*dz <= tol, "{name}: max |dz| = {dz:e} > {tol:e}");
    }
}

#[test]
fn reject_cases_are_errors_and_the_model_survives_them() {
    let (m, g) = (load_cpu(), golden());
    let rejects = g["reject"].as_array().unwrap();
    assert_eq!(rejects.len(), 4);
    for r in rejects {
        let ids = ids_of(&r["ids"]);
        assert!(
            m.forward(&ids).is_err(),
            "{} (L = {}) должен быть отклонён",
            r["name"],
            ids.len()
        );
    }
    // граница: pad_id + L < context_length, то есть L = 512 допустим (вне входного контракта 257…512, но не ошибка), 513 — нет
    let long = |n: usize| -> Vec<u32> {
        std::iter::once(0)
            .chain((1..n as u32 - 1).map(|i| 3 + i % 200))
            .chain(std::iter::once(2))
            .collect()
    };
    assert!(m.forward(&long(512)).is_ok());
    assert!(m.forward(&long(513)).is_err());
    assert!(m.forward(&[]).is_err());
    assert!(m.forward(&[0, 256, 2]).is_err(), "id = vocab_size");
    assert!(m.forward(&[0, u32::MAX, 2]).is_err());
    assert!(
        m.forward(&[0]).is_ok(),
        "L = 1 не ошибка: чтение не выходит за таблицы"
    );
    // после отказов модель считает по-прежнему
    for (name, dz) in run_golden(&m, &g) {
        assert!(dz <= tolerance(&g), "{name}: {dz:e}");
    }
}

#[test]
fn answer_does_not_depend_on_the_thread_pool() {
    let (m, g) = (load_cpu(), golden());
    let ids = ids_of(&g["cases"][12]["ids"]); // len_256
    let want = m.forward(&ids).unwrap(); // глобальный пул
    for n in [1, 2, 3, 4, 6] {
        let got = pool(n).install(|| m.forward(&ids)).unwrap();
        for (a, b) in got.iter().flatten().zip(want.iter().flatten()) {
            assert!((a - b).abs() < 1e-5, "пул из {n} потоков: {a} против {b}");
        }
    }
}

#[test]
fn parallel_callers_share_one_model() {
    let (m, g) = (Arc::new(load_cpu()), golden());
    let cases: Vec<(Vec<u32>, Vec<Vec<f64>>)> = g["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (ids_of(&c["ids"]), expected(c, m.outputs())))
        .collect();
    let cases = Arc::new(cases);
    let tol = tolerance(&g);
    let threads: Vec<_> = (0..4)
        .map(|t| {
            let (m, cases) = (m.clone(), cases.clone());
            std::thread::spawn(move || {
                for round in 0..3 {
                    for (i, (ids, want)) in cases.iter().enumerate() {
                        if (i + t + round) % 2 == 0 {
                            continue;
                        }
                        let got = m.forward(ids).unwrap();
                        assert!(max_dz(&got, want) <= tol, "поток {t}, случай {i}");
                    }
                }
            })
        })
        .collect();
    for th in threads {
        th.join().unwrap();
    }
}

// ------------------------------------------------------------------------------------------------ строгость читалки

fn find(hay: &[u8], needle: &[u8]) -> usize {
    hay.windows(needle.len())
        .position(|w| w == needle)
        .unwrap_or_else(|| panic!("в файле нет {:?}", String::from_utf8_lossy(needle)))
}

/// Значение u32-ключа лежит сразу за именем: тип (u32 = 4) и 4 байта значения.
fn patch_u32(bytes: &mut [u8], key: &str, v: u32) {
    let p = find(bytes, key.as_bytes()) + key.len();
    assert_eq!(
        u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()),
        4,
        "{key}: не u32"
    );
    bytes[p + 4..p + 8].copy_from_slice(&v.to_le_bytes());
}

fn patch_f32(bytes: &mut [u8], key: &str, v: f32) {
    let p = find(bytes, key.as_bytes()) + key.len();
    assert_eq!(
        u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()),
        6,
        "{key}: не f32"
    );
    bytes[p + 4..p + 8].copy_from_slice(&v.to_le_bytes());
}

/// Строка той же длины: тип (string = 8), длина u64, байты.
fn patch_str(bytes: &mut [u8], key: &str, old: &str, new: &str) {
    assert_eq!(old.len(), new.len());
    let p = find(bytes, key.as_bytes()) + key.len();
    assert_eq!(
        u32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()),
        8,
        "{key}: не строка"
    );
    assert_eq!(&bytes[p + 12..p + 12 + old.len()], old.as_bytes());
    bytes[p + 12..p + 12 + new.len()].copy_from_slice(new.as_bytes());
}

fn write_tmp(tag: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "yttri-decider-tiny-{}-{tag}.gguf",
        std::process::id()
    ));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn original() -> Vec<u8> {
    std::fs::read(dir().join("model.gguf")).unwrap()
}

fn load_err(tag: &str, bytes: &[u8]) -> String {
    let p = write_tmp(tag, bytes);
    let r = YttriDecider::load(&p, &Device::Cpu);
    let _ = std::fs::remove_file(&p);
    match r {
        Ok(_) => panic!("{tag}: испорченный файл загрузился"),
        Err(e) => format!("{e}"),
    }
}

#[test]
fn reader_rejects_a_broken_contract() {
    let ok = original();
    let mut b = ok.clone();
    patch_str(
        &mut b,
        "general.architecture",
        "yttri-decider",
        "yttri-decidex",
    );
    assert!(load_err("arch", &b).contains("general.architecture"));

    let mut b = ok.clone();
    patch_u32(&mut b, "yttri-decider.layout", 2);
    assert!(load_err("layout", &b).contains("раскладка"));

    let mut b = ok.clone();
    patch_str(&mut b, "yttri-decider.quant", "q8_0", "q4_0");
    assert!(load_err("quant", &b).contains("рецепт"));

    let mut b = ok.clone();
    patch_str(&mut b, "yttri-decider.pooling", "cls", "max");
    assert!(load_err("pooling", &b).contains("pooling"));

    let mut b = ok.clone();
    patch_str(&mut b, "yttri-decider.hidden_act", "gelu_erf", "gelu_new");
    assert!(load_err("act", &b).contains("hidden_act"));

    // 514 строк позиций, pad_id 1: max_tokens ≤ 512
    let mut b = ok.clone();
    patch_u32(&mut b, "yttri-decider.max_tokens", 513);
    assert!(load_err("max_tokens", &b).contains("max_tokens"));

    let mut b = ok.clone();
    patch_u32(&mut b, "yttri-decider.attention.head_count", 3);
    assert!(load_err("head_count", &b).contains("head_count"));

    let mut b = ok.clone();
    patch_u32(&mut b, "yttri-decider.pad_token_id", 256);
    assert!(load_err("pad", &b).contains("pad_token_id"));

    let mut b = ok.clone();
    patch_f32(&mut b, "yttri-decider.attention.layer_norm_epsilon", 0.0);
    assert!(load_err("eps", &b).contains("epsilon"));

    // ширина головы в KV расходится с формой out_proj
    let mut b = ok.clone();
    patch_u32(&mut b, "yttri-decider.head.mail_category", 8);
    assert!(load_err("head_width", &b).contains("mail_category"));

    // KV типа i32 вместо u32 (candle читает строго по типу)
    let mut b = ok.clone();
    let p = find(&b, b"yttri-decider.block_count") + "yttri-decider.block_count".len();
    b[p..p + 4].copy_from_slice(&5u32.to_le_bytes()); // GGUF i32
    assert!(load_err("i32", &b).contains("block_count"));

    // матрица энкодера не Q8_0
    let mut b = ok.clone();
    let name = b"encoder.layer.0.intermediate.dense.weight";
    let p = find(&b, name) + name.len() + 4 + 8 * 2; // n_dims (u32) и два измерения (u64) до типа тензора
    b[p..p + 4].copy_from_slice(&0u32.to_le_bytes()); // F32
    assert!(load_err("dtype", &b).contains("intermediate.dense.weight"));

    // обрезанный файл: данные тензоров выходят за пределы
    assert!(load_err("truncated", &ok[..ok.len() - 4096]).contains("пределы"));

    // отсутствующий файл
    assert!(YttriDecider::load(dir().join("нет-такого.gguf"), &Device::Cpu).is_err());
}

#[test]
fn eps_comes_from_the_file() {
    // боевой eps 1e-5 в той же модели: каждый из 14 входов обязан разойтись с golden заметно больше допуска (§10.9: ×28)
    let (g, tol) = (golden(), tolerance(&golden()));
    let mut b = original();
    patch_f32(&mut b, "yttri-decider.attention.layer_norm_epsilon", 1e-5);
    let p = write_tmp("eps-1e-5", &b);
    let m = YttriDecider::load(&p, &Device::Cpu).unwrap();
    let _ = std::fs::remove_file(&p);
    assert!((m.config().eps - 1e-5).abs() < 1e-12);
    for (name, dz) in run_golden(&m, &g) {
        assert!(
            dz > 10.0 * tol,
            "{name}: eps 1e-5 отличается от golden лишь на {dz:e}"
        );
    }
}

// ------------------------------------------------------------------------------------------------ эталон-оракул

/// Независимый naive-эталон в f64 по §10.6: читает те же байты файла (деквантизация candle, не наша), позиции HF.
/// Нужен только для проверки самого golden: мутации семантики должны уходить за допуск.
mod oracle {
    use super::*;

    pub struct Weights {
        t: HashMap<String, Vec<f64>>,
        hid: usize,
        nh: usize,
        layers: usize,
        eps: f64,
        pad: u32,
        heads: Vec<String>,
    }

    #[derive(Clone, Copy, Default)]
    pub struct Variant {
        pub eps: Option<f64>,
        pub gelu_tanh: bool,
        pub index_positions: bool,
        pub mask_pad: bool,
    }

    pub fn load(path: &Path) -> Result<Weights> {
        let mut f = std::fs::File::open(path)?;
        let c = Content::read(&mut f)?;
        let u = |k: &str| c.metadata[&format!("yttri-decider.{k}")].to_u32().unwrap() as usize;
        let mut t = HashMap::new();
        for name in c.tensor_infos.keys() {
            let q = c.tensor(&mut f, name, &Device::Cpu)?;
            let v = q
                .dequantize(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            t.insert(name.clone(), v.into_iter().map(f64::from).collect());
        }
        let heads = c.metadata["yttri-decider.heads"]
            .to_string()
            .unwrap()
            .split(',')
            .map(String::from)
            .collect();
        Ok(Weights {
            t,
            hid: u("embedding_length"),
            nh: u("attention.head_count"),
            layers: u("block_count"),
            eps: c.metadata["yttri-decider.attention.layer_norm_epsilon"]
                .to_f32()
                .unwrap() as f64,
            pad: u("pad_token_id") as u32,
            heads,
        })
    }

    /// erf в double: ряд Тейлора до |x| < 2,5, дальше цепная дробь для erfc.
    fn erf(x: f64) -> f64 {
        let ax = x.abs();
        let v = if ax < 2.5 {
            let (mut term, mut sum) = (ax, ax);
            for n in 1..80 {
                term *= -ax * ax / n as f64;
                sum += term / (2 * n + 1) as f64;
            }
            sum * 2.0 / std::f64::consts::PI.sqrt()
        } else {
            let mut t = ax;
            for n in (1..=80).rev() {
                t = ax + (n as f64 / 2.0) / t;
            }
            1.0 - (-ax * ax).exp() / (std::f64::consts::PI.sqrt() * t)
        };
        v.copysign(x)
    }

    fn linear(w: &[f64], b: &[f64], x: &[f64]) -> Vec<f64> {
        let k = x.len();
        (0..b.len())
            .map(|j| {
                b[j] + w[j * k..(j + 1) * k]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
            })
            .collect()
    }

    fn layer_norm(x: &[f64], g: &[f64], b: &[f64], eps: f64) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
        x.iter()
            .zip(g)
            .zip(b)
            .map(|((v, g), b)| (v - mean) / (var + eps).sqrt() * g + b)
            .collect()
    }

    fn gelu(x: f64, tanh: bool) -> f64 {
        if tanh {
            0.5 * x
                * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
        } else {
            0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
        }
    }

    pub fn forward(w: &Weights, ids: &[u32], v: Variant) -> Vec<Vec<f64>> {
        let (hid, nh) = (w.hid, w.nh);
        let hd = hid / nh;
        let eps = v.eps.unwrap_or(w.eps);
        let t = |name: &str| &w.t[name];
        let mut notpad = 0usize;
        let mut h: Vec<Vec<f64>> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let pos = if v.index_positions {
                    w.pad as usize + 1 + i
                } else if id == w.pad {
                    w.pad as usize
                } else {
                    notpad += 1;
                    notpad + w.pad as usize
                };
                let e = &t("embeddings.word_embeddings.weight")
                    [id as usize * hid..(id as usize + 1) * hid];
                let p = &t("embeddings.position_embeddings.weight")[pos * hid..(pos + 1) * hid];
                let tt = t("embeddings.token_type_embeddings.weight");
                let x: Vec<f64> = (0..hid).map(|j| e[j] + tt[j] + p[j]).collect();
                layer_norm(
                    &x,
                    t("embeddings.LayerNorm.weight"),
                    t("embeddings.LayerNorm.bias"),
                    eps,
                )
            })
            .collect();
        for li in 0..w.layers {
            let p = |s: &str, k: &str| format!("encoder.layer.{li}.{s}.{k}");
            let lin = |s: &str, x: &[f64]| linear(t(&p(s, "weight")), t(&p(s, "bias")), x);
            let q: Vec<Vec<f64>> = h.iter().map(|x| lin("attention.self.query", x)).collect();
            let k: Vec<Vec<f64>> = h.iter().map(|x| lin("attention.self.key", x)).collect();
            let vv: Vec<Vec<f64>> = h.iter().map(|x| lin("attention.self.value", x)).collect();
            let mut next = Vec::with_capacity(h.len());
            for i in 0..h.len() {
                let mut ctx = vec![0.0; hid];
                for head in 0..nh {
                    let o = head * hd;
                    let mut s: Vec<f64> = (0..h.len())
                        .map(|j| {
                            if v.mask_pad && ids[j] == w.pad {
                                f64::NEG_INFINITY
                            } else {
                                (0..hd).map(|d| q[i][o + d] * k[j][o + d]).sum::<f64>()
                                    / (hd as f64).sqrt()
                            }
                        })
                        .collect();
                    let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    s.iter_mut().for_each(|x| *x = (*x - m).exp());
                    let z: f64 = s.iter().sum();
                    for (j, pj) in s.iter().enumerate() {
                        for d in 0..hd {
                            ctx[o + d] += pj / z * vv[j][o + d];
                        }
                    }
                }
                let ao = lin("attention.output.dense", &ctx);
                let x: Vec<f64> = ao.iter().zip(&h[i]).map(|(a, b)| a + b).collect();
                let a = layer_norm(
                    &x,
                    t(&p("attention.output.LayerNorm", "weight")),
                    t(&p("attention.output.LayerNorm", "bias")),
                    eps,
                );
                let f: Vec<f64> = lin("intermediate.dense", &a)
                    .into_iter()
                    .map(|x| gelu(x, v.gelu_tanh))
                    .collect();
                let f2 = lin("output.dense", &f);
                let x: Vec<f64> = f2.iter().zip(&a).map(|(a, b)| a + b).collect();
                next.push(layer_norm(
                    &x,
                    t(&p("output.LayerNorm", "weight")),
                    t(&p("output.LayerNorm", "bias")),
                    eps,
                ));
            }
            h = next;
        }
        w.heads
            .iter()
            .map(|q| {
                let z: Vec<f64> = linear(
                    t(&format!("heads.{q}.dense.weight")),
                    t(&format!("heads.{q}.dense.bias")),
                    &h[0],
                )
                .into_iter()
                .map(f64::tanh)
                .collect();
                linear(
                    t(&format!("heads.{q}.out_proj.weight")),
                    t(&format!("heads.{q}.out_proj.bias")),
                    &z,
                )
            })
            .collect()
    }
}

fn oracle_dz(
    w: &oracle::Weights,
    g: &Value,
    outputs: &[(String, usize)],
    v: oracle::Variant,
) -> Vec<(String, bool, f64)> {
    g["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            let ids = ids_of(&c["ids"]);
            let got: Vec<Vec<f32>> = oracle::forward(w, &ids, v)
                .into_iter()
                .map(|z| z.into_iter().map(|x| x as f32).collect())
                .collect();
            let has_pad = ids.contains(&1);
            (
                c["name"].as_str().unwrap().to_string(),
                has_pad,
                max_dz(&got, &expected(c, outputs)),
            )
        })
        .collect()
}

#[test]
fn golden_is_sensitive_to_eps_gelu_positions_and_pad_mask() {
    let (g, m) = (golden(), load_cpu());
    let tol = tolerance(&g);
    let w = oracle::load(&dir().join("model.gguf")).unwrap();
    let base = oracle::Variant::default();

    // сам оракул в допуске: иначе его мутации ничего не доказывают
    let ok = oracle_dz(&w, &g, m.outputs(), base);
    eprintln!(
        "оракул f64, §10.6: {:?}",
        ok.iter()
            .map(|r| format!("{:.1e}", r.2))
            .collect::<Vec<_>>()
    );
    for (name, _, dz) in &ok {
        assert!(*dz <= tol / 10.0, "оракул {name}: {dz:e}");
    }

    // боевой eps вместо eps файла: ≥ ×10 допуска на каждом входе
    let eps = oracle_dz(
        &w,
        &g,
        m.outputs(),
        oracle::Variant {
            eps: Some(1e-5),
            ..base
        },
    );
    eprintln!(
        "мутант eps = 1e-5: min dz = {:.2e}",
        eps.iter().map(|r| r.2).fold(f64::MAX, f64::min)
    );
    for (name, _, dz) in &eps {
        assert!(
            *dz > 10.0 * tol,
            "eps 1e-5 не ловится golden на {name}: {dz:e}"
        );
    }
    // eps HF по умолчанию (1e-12) — тоже
    let eps = oracle_dz(
        &w,
        &g,
        m.outputs(),
        oracle::Variant {
            eps: Some(1e-12),
            ..base
        },
    );
    for (name, _, dz) in &eps {
        assert!(
            *dz > 10.0 * tol,
            "eps 1e-12 не ловится golden на {name}: {dz:e}"
        );
    }
    // tanh-GELU вместо erf: за допуск на каждом входе
    let gelu = oracle_dz(
        &w,
        &g,
        m.outputs(),
        oracle::Variant {
            gelu_tanh: true,
            ..base
        },
    );
    eprintln!(
        "мутант tanh-GELU: min dz = {:.2e}",
        gelu.iter().map(|r| r.2).fold(f64::MAX, f64::min)
    );
    for (name, _, dz) in &gelu {
        assert!(
            *dz > 1.5 * tol,
            "tanh-GELU не ловится golden на {name}: {dz:e}"
        );
    }
    // индексные позиции pad_id + 1 + i и маска по ids != pad: различаются только на входах с литералом <pad>
    for (label, v) in [
        (
            "индексные позиции",
            oracle::Variant {
                index_positions: true,
                ..base
            },
        ),
        (
            "маска по <pad>",
            oracle::Variant {
                mask_pad: true,
                ..base
            },
        ),
    ] {
        let rows = oracle_dz(&w, &g, m.outputs(), v);
        let with_pad: Vec<_> = rows.iter().filter(|r| r.1).collect();
        assert!(
            with_pad.len() >= 5,
            "в golden должно быть не меньше пяти входов с <pad>: {}",
            with_pad.len()
        );
        eprintln!(
            "мутант {label}: min dz на входах с <pad> = {:.2e}",
            with_pad.iter().map(|r| r.2).fold(f64::MAX, f64::min)
        );
        for (name, _, dz) in &with_pad {
            assert!(
                *dz > 100.0 * tol,
                "{label} не ловится golden на {name}: {dz:e}"
            );
        }
        for (name, _, dz) in rows.iter().filter(|r| !r.1) {
            assert!(
                *dz <= tol,
                "{label}: вход без <pad> {name} не должен меняться: {dz:e}"
            );
        }
    }
}

/// Случайные входы (в том числе с литералом `<pad>` и длиннее `max_tokens`, но в пределах таблицы позиций): lean на обоих
/// наборах ядер против независимого оракула. Golden покрывает 14 входов, здесь — ещё 60 разной длины.
#[test]
fn random_inputs_match_the_oracle() {
    let w = oracle::load(&dir().join("model.gguf")).unwrap();
    let best = load_cpu();
    let base =
        YttriDecider::load_with_baseline_kernels(dir().join("model.gguf"), &Device::Cpu).unwrap();
    let mut seed = 0x5eed_u64;
    let mut next = move || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as u32
    };
    let mut lengths: Vec<usize> = vec![
        1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 257, 300, 512,
    ];
    lengths.extend((0..43).map(|_| 2 + next() as usize % 200));
    let mut worst = 0f64;
    for (n, &l) in lengths.iter().enumerate() {
        let mut ids: Vec<u32> = (0..l).map(|_| next() % 256).collect();
        if l >= 2 {
            ids[0] = 0;
            ids[l - 1] = 2;
        }
        if n % 3 == 0 {
            // литерал <pad> внутри входа
            for _ in 0..1 + l / 40 {
                let i = next() as usize % l;
                ids[i] = 1;
            }
        }
        let want = oracle::forward(&w, &ids, oracle::Variant::default());
        for (label, m) in [("best", &best), ("baseline", &base)] {
            let got = m.forward(&ids).unwrap();
            let dz = got
                .iter()
                .flatten()
                .zip(want.iter().flatten())
                .map(|(a, b)| (*a as f64 - b).abs())
                .fold(0.0, f64::max);
            worst = worst.max(dz);
            assert!(dz < 2e-5, "{label}: L = {l}: max |dz| = {dz:e}");
        }
    }
    eprintln!("60 случайных входов против оракула: max |dz| = {worst:.2e}");
}

// ------------------------------------------------------------------------------------------------ CUDA

/// Тензорный путь на GPU против golden и против CPU-пути. Без CUDA-устройства тест пропускается громко.
#[cfg(feature = "cuda")]
#[test]
fn cuda_matches_the_reference_and_cpu() {
    let dev = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED cuda_matches_the_reference_and_cpu: устройства CUDA нет ({e})");
            return;
        }
    };
    let (g, tol) = (golden(), tolerance(&golden()));
    let gpu = YttriDecider::load(dir().join("model.gguf"), &dev).unwrap();
    let cpu = load_cpu();
    assert_eq!(gpu.outputs(), cpu.outputs());
    let rows = run_golden(&gpu, &g);
    let worst = rows.iter().map(|r| r.1).fold(0.0, f64::max);
    eprintln!("CUDA: max |dz| против golden = {worst:.3e}; по случаям {rows:?}");
    for (name, dz) in &rows {
        assert!(*dz <= tol, "CUDA {name}: {dz:e} > {tol:e}");
    }
    // GPU против CPU на тех же входах: оба f32, расхождение — порядок суммирования
    let mut cross = 0f32;
    for c in g["cases"].as_array().unwrap() {
        let ids = ids_of(&c["ids"]);
        let (a, b) = (gpu.forward(&ids).unwrap(), cpu.forward(&ids).unwrap());
        for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
            cross = cross.max((x - y).abs());
        }
    }
    eprintln!("CUDA против CPU: max |dz| = {cross:.3e}");
    assert!(cross <= 1e-4);
    // отказы те же, что у lean
    for r in g["reject"].as_array().unwrap() {
        assert!(gpu.forward(&ids_of(&r["ids"])).is_err(), "{}", r["name"]);
    }
    assert!(gpu.resident_bytes() > 0);
}
