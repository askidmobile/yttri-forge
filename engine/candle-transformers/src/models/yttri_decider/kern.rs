//! Ядра CPU-пути decider: exp / erf / GELU / softmax / LayerNorm / деквантизация Q8_0 (f32).
//!
//! Без зависимостей и без `target-cpu`: тело ядра пишется один раз (`#[inline(always)]`) и инстанцируется
//! дважды — базовый вариант (SSE2 на x86-64, NEON на aarch64, скаляр на остальных) и AVX2+FMA под
//! `#[target_feature]` с проверкой `is_x86_feature_detected!` в рантайме. Так продуктовая сборка (релиз Yttri
//! идёт без `-C target-cpu`) получает AVX2 там, где он есть, и работает там, где его нет.
//!
//! Все ядра однопоточные: параллелит вызывающий (по строкам), в пуле rayon, который ему передали.

// коэффициенты Cephes и Abramowitz–Stegun приведены в исходной точности
#![allow(clippy::excessive_precision)]

use std::sync::OnceLock;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2_HI: f32 = 0.693_359_375;
const LN2_LO: f32 = -2.121_944_4e-4;
const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: округление до целого сложением
const FRAC_1_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

/// Набор инструкций, под который скомпилирован вариант ядра.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Isa {
    /// Базовый уровень целевой платформы (SSE2 на x86-64).
    Base,
    /// AVX2 + FMA (только x86-64, определяется в рантайме).
    Avx2,
}

pub(crate) fn has_avx2() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

pub(crate) fn best_isa() -> Isa {
    if has_avx2() {
        Isa::Avx2
    } else {
        Isa::Base
    }
}

#[inline(always)]
fn fma<const F: bool>(a: f32, b: f32, c: f32) -> f32 {
    // F = true только внутри функций с `target_feature = "fma"`: иначе `mul_add` уходит в медленный libm
    if F {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// exp(x) для f32: приведение к 2^n * e^r, полином Cephes `expf` (относительная погрешность ~1e-7).
#[inline(always)]
pub(crate) fn exp_poly<const F: bool>(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let t = x * LOG2E + MAGIC;
    let n = t - MAGIC;
    let r = fma::<F>(-n, LN2_LO, fma::<F>(-n, LN2_HI, x));
    let mut p = 1.987_569_15e-4_f32;
    p = fma::<F>(p, r, 1.398_199_950_7e-3);
    p = fma::<F>(p, r, 8.333_451_907_3e-3);
    p = fma::<F>(p, r, 4.166_579_589_4e-2);
    p = fma::<F>(p, r, 1.666_666_545_9e-1);
    p = fma::<F>(p, r, 5.000_000_120_1e-1);
    let e = fma::<F>(p * r, r, r) + 1.0;
    let bits = ((t.to_bits().wrapping_sub(MAGIC.to_bits())) as i32).wrapping_add(127) << 23;
    e * f32::from_bits(bits as u32)
}

/// erf(v/√2) по Abramowitz–Stegun 7.1.26 (|Δ| ≤ 1.5e-7 в double; в f32 замерено ≤ 5.8e-7).
#[inline(always)]
pub(crate) fn erf_half_poly<const F: bool>(v: f32) -> f32 {
    let a = v.abs() * FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let mut p = 1.061_405_429_f32;
    p = fma::<F>(p, t, -1.453_152_027);
    p = fma::<F>(p, t, 1.421_413_741);
    p = fma::<F>(p, t, -0.284_496_736);
    p = fma::<F>(p, t, 0.254_829_592);
    p *= t;
    let e = exp_poly::<F>(-(a * a));
    let erf_abs = 1.0 - p * e;
    f32::from_bits((erf_abs.to_bits() & 0x7fff_ffff) | (v.to_bits() & 0x8000_0000))
}

/// Точный GELU (erf), не tanh-приближение: `0,5·v·(1 + erf(v/√2))`.
#[inline(always)]
pub(crate) fn gelu_poly<const F: bool>(v: f32) -> f32 {
    0.5 * v * (1.0 + erf_half_poly::<F>(v))
}

// ---------------------------------------------------------------- GELU + bias

#[inline(always)]
fn gelu_bias_impl<const F: bool>(x: &mut [f32], bias: &[f32], n: usize) {
    for row in x.chunks_exact_mut(n) {
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v = gelu_poly::<F>(*v + b);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gelu_bias_avx2(x: &mut [f32], bias: &[f32], n: usize) {
    gelu_bias_impl::<true>(x, bias, n)
}

/// x[строка] = gelu(x[строка] + bias), на месте.
pub(crate) fn gelu_bias(isa: Isa, x: &mut [f32], bias: &[f32], n: usize) {
    match isa {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: Isa::Avx2 выдаёт только best_isa() после is_x86_feature_detected!
        Isa::Avx2 => unsafe { gelu_bias_avx2(x, bias, n) },
        _ => gelu_bias_impl::<false>(x, bias, n),
    }
}

// ---------------------------------------------------------------- bias

#[inline(always)]
fn bias_impl(x: &mut [f32], bias: &[f32], n: usize) {
    for row in x.chunks_exact_mut(n) {
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bias_avx2(x: &mut [f32], bias: &[f32], n: usize) {
    bias_impl(x, bias, n)
}

/// x[строка] += bias, на месте.
pub(crate) fn add_bias(isa: Isa, x: &mut [f32], bias: &[f32], n: usize) {
    match isa {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: см. gelu_bias
        Isa::Avx2 => unsafe { bias_avx2(x, bias, n) },
        _ => bias_impl(x, bias, n),
    }
}

// ---------------------------------------------------------------- softmax

#[inline(always)]
fn max_lanes(row: &[f32]) -> f32 {
    let body = row.len() / 8 * 8;
    let mut ml = [f32::NEG_INFINITY; 8];
    for c in row[..body].chunks_exact(8) {
        for i in 0..8 {
            ml[i] = if c[i] > ml[i] { c[i] } else { ml[i] };
        }
    }
    let mut m = f32::NEG_INFINITY;
    for &v in &row[body..] {
        if v > m {
            m = v;
        }
    }
    for v in ml {
        if v > m {
            m = v;
        }
    }
    m
}

#[inline(always)]
fn sum_lanes(row: &[f32]) -> f32 {
    let body = row.len() / 8 * 8;
    let mut sl = [0f32; 8];
    for c in row[..body].chunks_exact(8) {
        for i in 0..8 {
            sl[i] += c[i];
        }
    }
    let mut tail = 0f32;
    for &v in &row[body..] {
        tail += v;
    }
    ((sl[0] + sl[4]) + (sl[1] + sl[5])) + ((sl[2] + sl[6]) + (sl[3] + sl[7])) + tail
}

// Циклы разделены намеренно: чистый map (exp) векторизуется надёжно, а смешанный «exp + накопление суммы» — нет.
#[inline(always)]
fn softmax_impl<const F: bool>(x: &mut [f32], n: usize) {
    for row in x.chunks_exact_mut(n) {
        let m = max_lanes(row);
        for v in row.iter_mut() {
            *v = exp_poly::<F>(*v - m);
        }
        let inv = 1.0 / sum_lanes(row);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn softmax_avx2(x: &mut [f32], n: usize) {
    softmax_impl::<true>(x, n)
}

/// Softmax по строкам длины n, на месте.
pub(crate) fn softmax_rows(isa: Isa, x: &mut [f32], n: usize) {
    match isa {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: см. gelu_bias
        Isa::Avx2 => unsafe { softmax_avx2(x, n) },
        _ => softmax_impl::<false>(x, n),
    }
}

// ---------------------------------------------------------------- LayerNorm

/// x[строка] = LN(x[строка] + bias? + resid?) · g + b, смещённая дисперсия, двухпроходная (как torch).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ln_impl(
    x: &mut [f32],
    resid: Option<&[f32]>,
    bias: Option<&[f32]>,
    g: &[f32],
    b: &[f32],
    n: usize,
    eps: f32,
) {
    for (ri, row) in x.chunks_exact_mut(n).enumerate() {
        if let Some(bias) = bias {
            for (v, &bb) in row.iter_mut().zip(bias.iter()) {
                *v += bb;
            }
        }
        if let Some(res) = resid {
            let rr = &res[ri * n..(ri + 1) * n];
            for (v, &r) in row.iter_mut().zip(rr.iter()) {
                *v += r;
            }
        }
        let mean = sum_lanes(row) / n as f32;
        let body = n / 8 * 8;
        let mut vl = [0f32; 8];
        for c in row[..body].chunks_exact(8) {
            for i in 0..8 {
                let d = c[i] - mean;
                vl[i] += d * d;
            }
        }
        let mut vs = ((vl[0] + vl[4]) + (vl[1] + vl[5])) + ((vl[2] + vl[6]) + (vl[3] + vl[7]));
        for &v in &row[body..] {
            let d = v - mean;
            vs += d * d;
        }
        let inv = 1.0 / (vs / n as f32 + eps).sqrt();
        for ((v, &gg), &bb) in row.iter_mut().zip(g.iter()).zip(b.iter()) {
            *v = (*v - mean) * inv * gg + bb;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn ln_avx2(
    x: &mut [f32],
    resid: Option<&[f32]>,
    bias: Option<&[f32]>,
    g: &[f32],
    b: &[f32],
    n: usize,
    eps: f32,
) {
    ln_impl(x, resid, bias, g, b, n, eps)
}

/// LayerNorm по строкам длины n с необязательными bias и остаточной связью, на месте.
#[allow(clippy::too_many_arguments)]
pub(crate) fn layer_norm_rows(
    isa: Isa,
    x: &mut [f32],
    resid: Option<&[f32]>,
    bias: Option<&[f32]>,
    g: &[f32],
    b: &[f32],
    n: usize,
    eps: f32,
) {
    match isa {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: см. gelu_bias
        Isa::Avx2 => unsafe { ln_avx2(x, resid, bias, g, b, n, eps) },
        _ => ln_impl(x, resid, bias, g, b, n, eps),
    }
}

// ---------------------------------------------------------------- Q8_0

/// IEEE half → f32 (в том числе субнормальные: масштаб блока `amax/127` у малых весов может в них попасть).
#[inline]
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    match exp {
        0 => {
            // субнормальное: man · 2^-24, точно представимо в f32
            let v = man as f32 * (1.0 / 16_777_216.0);
            if sign != 0 {
                -v
            } else {
                v
            }
        }
        31 => f32::from_bits(sign | 0x7f80_0000 | (man << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (man << 13)),
    }
}

/// Деквантизация строки Q8_0 (блоки по 34 байта: f16 `d` + 32 × i8) в f32: значение = `d · q`, одно умножение f32.
#[inline]
pub(crate) fn dequant_q8_0_row(src: &[u8], out: &mut [f32]) {
    for (blk, o) in src.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for (v, &q) in o.iter_mut().zip(blk[2..].iter()) {
            *v = d * (q as i8) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// erf в double: ряд Тейлора до |x| < 2,5, дальше цепная дробь для erfc.
    fn erf64(x: f64) -> f64 {
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

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn reference_erf_is_sane() {
        // известные значения: erf(0,5) = 0,5204998778130465, erf(3) = 0,9999779095030014
        assert!((erf64(0.5) - 0.520_499_877_813_046_5).abs() < 1e-14);
        assert!((erf64(3.0) - 0.999_977_909_503_001_4).abs() < 1e-14);
        assert!((erf64(-1.0) + 0.842_700_792_949_714_9).abs() < 1e-14);
    }

    #[test]
    fn f16_conversion() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // наименьшее субнормальное
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * (1.0 / 16_777_216.0)); // наибольшее субнормальное
        assert_eq!(f16_to_f32(0x0400), 6.103_515_6e-5); // наименьшее нормальное
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
        // все 65536 значений против arithmetic: (-1)^s · 2^(e-15) · (1 + m/1024)
        for h in 0..=u16::MAX {
            let (s, e, m) = ((h >> 15) & 1, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f64);
            if e == 31 {
                continue;
            }
            let mag = if e == 0 {
                m * 2f64.powi(-24)
            } else {
                (1.0 + m / 1024.0) * 2f64.powi(e - 15)
            };
            let want = if s == 1 { -mag } else { mag } as f32;
            assert_eq!(f16_to_f32(h).to_bits(), want.to_bits(), "half {h:#06x}");
        }
    }

    #[test]
    fn gelu_matches_double_erf() {
        let mut seed = 1u64;
        let (mut max_erf, mut max_gelu) = (0f64, 0f64);
        for i in 0..400_000 {
            // плотная сетка около нуля и редкие большие значения (до ±30: хвост erf → ±1)
            let v = if i % 4 == 0 {
                (lcg(&mut seed) * 60.0 - 30.0) as f32
            } else {
                (lcg(&mut seed) * 16.0 - 8.0) as f32
            };
            let e = erf_half_poly::<false>(v) as f64;
            let want = erf64(v as f64 / std::f64::consts::SQRT_2);
            max_erf = max_erf.max((e - want).abs());
            let g = gelu_poly::<false>(v) as f64;
            let gw = 0.5 * v as f64 * (1.0 + want);
            max_gelu = max_gelu.max((g - gw).abs() / (1.0 + (v as f64).abs()));
        }
        assert!(max_erf <= 1e-6, "erf: {max_erf:e}");
        assert!(max_gelu <= 1e-6, "gelu (относительно 1+|v|): {max_gelu:e}");
        // предельные значения: хвосты не дают NaN и стремятся к 0 и v
        assert_eq!(gelu_poly::<false>(-1e4), 0.0);
        assert_eq!(gelu_poly::<false>(1e4), 1e4);
        assert_eq!(gelu_poly::<false>(0.0), 0.0);
    }

    #[test]
    fn exp_poly_is_accurate() {
        let mut seed = 7u64;
        let mut max_rel = 0f64;
        for _ in 0..200_000 {
            let x = (lcg(&mut seed) * 87.0 - 87.0) as f32; // softmax видит только x ≤ 0
            let rel = ((exp_poly::<false>(x) as f64) / (x as f64).exp() - 1.0).abs();
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 5e-7, "{max_rel:e}");
    }

    fn rand_vec(n: usize, scale: f32, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| ((lcg(seed) * 2.0 - 1.0) as f32) * scale)
            .collect()
    }

    #[test]
    fn softmax_rows_match_reference() {
        let mut seed = 3u64;
        for &n in &[1usize, 2, 7, 8, 9, 31, 64, 255, 256, 257, 512] {
            let src = rand_vec(3 * n, 6.0, &mut seed);
            let mut got = src.clone();
            softmax_rows(Isa::Base, &mut got, n);
            for (r, row) in src.chunks_exact(n).enumerate() {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let e: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp()).collect();
                let s: f64 = e.iter().sum();
                for (j, ev) in e.iter().enumerate() {
                    assert!(
                        (got[r * n + j] as f64 - ev / s).abs() < 3e-7,
                        "n={n} r={r} j={j}"
                    );
                }
            }
        }
    }

    #[test]
    fn layer_norm_rows_match_reference() {
        let mut seed = 5u64;
        for &(n, eps) in &[(64usize, 1e-2f32), (384, 1e-5), (33, 1e-5)] {
            let rows = 5;
            let src = rand_vec(rows * n, 3.0, &mut seed);
            let resid = rand_vec(rows * n, 1.0, &mut seed);
            let bias = rand_vec(n, 0.5, &mut seed);
            let (g, b) = (rand_vec(n, 1.0, &mut seed), rand_vec(n, 1.0, &mut seed));
            let mut got = src.clone();
            layer_norm_rows(
                Isa::Base,
                &mut got,
                Some(&resid),
                Some(&bias),
                &g,
                &b,
                n,
                eps,
            );
            for r in 0..rows {
                let x: Vec<f64> = (0..n)
                    .map(|j| (src[r * n + j] + bias[j] + resid[r * n + j]) as f64)
                    .collect();
                let mean = x.iter().sum::<f64>() / n as f64;
                let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64;
                for j in 0..n {
                    let want =
                        (x[j] - mean) / (var + eps as f64).sqrt() * g[j] as f64 + b[j] as f64;
                    assert!(
                        (got[r * n + j] as f64 - want).abs() < 2e-5,
                        "n={n} r={r} j={j}: {} vs {want}",
                        got[r * n + j]
                    );
                }
            }
        }
    }

    #[test]
    fn avx2_and_base_variants_agree() {
        if !has_avx2() {
            eprintln!("avx2+fma недоступны на этой машине: проверка пропущена");
            return;
        }
        let mut seed = 9u64;
        let n = 384;
        let src = rand_vec(4 * n, 4.0, &mut seed);
        let bias = rand_vec(n, 0.5, &mut seed);
        let (g, b) = (rand_vec(n, 1.0, &mut seed), rand_vec(n, 1.0, &mut seed));
        let resid = rand_vec(4 * n, 1.0, &mut seed);
        let close = |a: &[f32], b: &[f32], tol: f32| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= tol * (1.0 + x.abs()))
        };
        let (mut a, mut c) = (src.clone(), src.clone());
        gelu_bias(Isa::Base, &mut a, &bias, n);
        gelu_bias(Isa::Avx2, &mut c, &bias, n);
        assert!(close(&a, &c, 1e-6));
        let (mut a, mut c) = (src.clone(), src.clone());
        softmax_rows(Isa::Base, &mut a, 96);
        softmax_rows(Isa::Avx2, &mut c, 96);
        assert!(close(&a, &c, 1e-6));
        let (mut a, mut c) = (src.clone(), src.clone());
        layer_norm_rows(
            Isa::Base,
            &mut a,
            Some(&resid),
            Some(&bias),
            &g,
            &b,
            n,
            1e-5,
        );
        layer_norm_rows(
            Isa::Avx2,
            &mut c,
            Some(&resid),
            Some(&bias),
            &g,
            &b,
            n,
            1e-5,
        );
        assert!(close(&a, &c, 1e-5));
    }

    #[test]
    fn dequant_row_matches_block_layout() {
        // два блока: d = 0,5 (0x3800) и d = 2 (0x4000)
        let mut src = Vec::new();
        for (d, base) in [(0x3800u16, 0i8), (0x4000u16, -16i8)] {
            src.extend_from_slice(&d.to_le_bytes());
            src.extend((0..32).map(|i| (base + i as i8) as u8));
        }
        let mut out = vec![0f32; 64];
        dequant_q8_0_row(&src, &mut out);
        for i in 0..32 {
            assert_eq!(out[i], 0.5 * i as f32);
            assert_eq!(out[32 + i], 2.0 * (-16 + i as i32) as f32);
        }
    }
}
