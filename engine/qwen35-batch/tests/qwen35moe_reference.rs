//! Phase 2 reference tests for Qwen3.6 35B-A3B MoE block.
//!
//! Synthetic experts with analytically known outputs:
//! - Identity SwiGLU expert: gate=up=down=I → output = silu(x) * x
//! - Zero expert: output = 0
//! - Shared expert gate 0 / 0.5 / 1 via gate_inp weight control
//! - Chunked vs unchunked reference tolerance
//! - Routing: deterministic top-k selection with known router weights

#![cfg(feature = "real-model")]

use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use qwen35_batch::real::moe::{
    f32_qtensor, ForwardMode, MoeBackend, MoeRouter, PackedExperts, Qwen35MoeBlock,
    Qwen35MoeConfig, SharedExpert,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Tolerance for F32 reference comparisons (no quantization, pure F32 matmul).
const TOL: f32 = 1e-5;

fn dev() -> Device {
    Device::Cpu
}

/// Identity matrix [n, n] as a flat F32 Tensor (row-major).
fn identity_flat(n: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

/// Create a 3D packed-expert tensor [n_experts, rows, cols] where every expert
/// is the identity matrix (requires rows == cols).
fn packed_identity(n_experts: usize, n: usize) -> Tensor {
    let rows_per_expert = n;
    let cols_per_expert = n;
    let total = n_experts * rows_per_expert * cols_per_expert;
    let mut data = vec![0.0f32; total];
    for e in 0..n_experts {
        for i in 0..n {
            let row = i;
            let col = i;
            let idx = e * (rows_per_expert * cols_per_expert) + row * cols_per_expert + col;
            data[idx] = 1.0;
        }
    }
    Tensor::from_vec(data, (n_experts, rows_per_expert, cols_per_expert), &dev()).unwrap()
}

/// Create a 3D packed-expert tensor [n_experts, rows, cols] of all zeros.
fn packed_zeros(n_experts: usize, rows: usize, cols: usize) -> Tensor {
    let total = n_experts * rows * cols;
    Tensor::from_vec(vec![0.0f32; total], (n_experts, rows, cols), &dev()).unwrap()
}

/// Build a Qwen35MoeBlock from raw tensors on CPU.
fn build_moe_block(
    n_embd: usize,
    n_ff: usize,
    n_experts: usize,
    n_experts_per_tok: usize,
    norm_topk: bool,
    router_weight: &Tensor,       // [n_experts, n_embd]
    packed_gate: &Tensor,         // [n_experts, n_ff, n_embd]
    packed_up: &Tensor,           // [n_experts, n_ff, n_embd]
    packed_down: &Tensor,         // [n_experts, n_embd, n_ff]
    shexp_gate_inp: &Tensor,      // [1, n_embd]
    shexp_gate: &Tensor,          // [n_ff, n_embd]
    shexp_up: &Tensor,            // [n_ff, n_embd]
    shexp_down: &Tensor,          // [n_embd, n_ff]
) -> Qwen35MoeBlock {
    let router = MoeRouter::new(
        Linear::new(router_weight.clone(), None),
        n_experts,
        n_experts_per_tok,
        norm_topk,
    );

    let routed = PackedExperts {
        gate: f32_qtensor(packed_gate).unwrap(),
        up: f32_qtensor(packed_up).unwrap(),
        down: f32_qtensor(packed_down).unwrap(),
        n_experts,
    };

    let shared = SharedExpert::new(
        Linear::new(shexp_gate_inp.clone(), None),
        candle_core::quantized::QMatMul::from_arc(f32_qtensor(shexp_gate).unwrap()).unwrap(),
        candle_core::quantized::QMatMul::from_arc(f32_qtensor(shexp_up).unwrap()).unwrap(),
        candle_core::quantized::QMatMul::from_arc(f32_qtensor(shexp_down).unwrap()).unwrap(),
    );

    let cfg = Qwen35MoeConfig {
        hidden_size: n_embd,
        n_experts,
        n_experts_per_tok,
        routed_intermediate: n_ff,
        shared_intermediate: n_ff,
        norm_topk_prob: norm_topk,
    };

    Qwen35MoeBlock::new(cfg, router, routed, shared, MoeBackend::Reference)
}

/// silu(x) * x computed on CPU for reference.
fn silu_mul_ref(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .map(|&x| {
            let s = x / (1.0 + (-x).exp());
            s * x
        })
        .collect()
}

/// Extract a 2D slice [rows, cols] from a 3D packed tensor at expert index `e`.
fn expert_slice(packed: &Tensor, e: usize, _rows: usize, _cols: usize) -> Tensor {
    packed.narrow(0, e, 1).unwrap().squeeze(0).unwrap()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

/// Identity SwiGLU expert: gate=up=down=I, routing weight 1.0, shared gate 0.
/// Expected: output = silu(x) * x.
#[test]
fn test_identity_routed_expert_silu_mul() {
    let n_embd = 4usize;
    let n_ff = 4usize; // == n_embd for identity
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize; // top-1

    // Router: token 0 → expert 0, token 1 → expert 1.
    // Router weight [n_experts, n_embd]: expert 0 = [1,0,0,0], expert 1 = [0,1,0,0].
    // For input x = [a, b, c, d]:
    //   logits[0] = a (dot with expert 0 weight)
    //   logits[1] = b (dot with expert 1 weight)
    // If a > b → expert 0 selected; if b > a → expert 1 selected.
    // We make both experts identity, so regardless of selection, output = silu(x)*x.
    let router_w = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 0.0, // expert 0
            0.0, 1.0, 0.0, 0.0, // expert 1
        ],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    let packed_gate = packed_identity(n_experts, n_embd); // [n_experts, n_ff, n_embd]
    let packed_up = packed_identity(n_experts, n_embd);
    let packed_down = packed_identity(n_experts, n_embd);

    // Shared expert: gate_inp = large negative → gate_scalar ≈ 0.
    let shexp_gate_inp = Tensor::from_vec(
        vec![-100.0f32; n_embd],
        (1, n_embd),
        &dev(),
    )
    .unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true, // norm_topk_prob
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    // Input: [1, 2, n_embd] — two tokens.
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
    let xs = Tensor::from_vec(x_data.clone(), (1, 2, n_embd), &dev()).unwrap();

    let out = block.forward(&xs, ForwardMode::Prefill).unwrap();
    let out_vec: Vec<f32> = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    // Expected: silu(x) * x for each element (shared gate ≈ 0).
    let expected = silu_mul_ref(&x_data);

    for (i, (got, exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "element {i}: got {got}, expected {exp} (silu(x)*x)"
        );
    }
}

/// Shared expert gate = 0.5 (gate_inp weight = 0 → silu_div(0) = 0.5).
/// Routed experts = zero, shared = identity SwiGLU.
/// Expected: output = 0.5 * silu(x) * x.
#[test]
fn test_shared_gate_half() {
    let n_embd = 4usize;
    let n_ff = 4usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize;

    // Router: any (routed experts are zero, so routing doesn't matter).
    let router_w = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    // Routed experts: all zeros.
    let packed_gate = packed_zeros(n_experts, n_ff, n_embd);
    let packed_up = packed_zeros(n_experts, n_ff, n_embd);
    let packed_down = packed_zeros(n_experts, n_embd, n_ff);

    // Shared expert: gate_inp = 0 → gate_scalar = 0.5.
    let shexp_gate_inp = Tensor::from_vec(vec![0.0f32; n_embd], (1, n_embd), &dev()).unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true,
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let xs = Tensor::from_vec(x_data.clone(), (1, 1, n_embd), &dev()).unwrap();

    let out = block.forward(&xs, ForwardMode::Prefill).unwrap();
    let out_vec: Vec<f32> = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    // Expected: 0.5 * silu(x) * x.
    let expected: Vec<f32> = silu_mul_ref(&x_data).iter().map(|v| v * 0.5).collect();

    for (i, (got, exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "element {i}: got {got}, expected {exp} (0.5 * silu(x)*x)"
        );
    }
}

/// Shared expert gate = 1.0 (gate_inp weight = large positive → silu_div → 1.0).
/// Routed experts = zero, shared = identity SwiGLU.
/// Expected: output = silu(x) * x.
#[test]
fn test_shared_gate_one() {
    let n_embd = 4usize;
    let n_ff = 4usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize;

    let router_w = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    let packed_gate = packed_zeros(n_experts, n_ff, n_embd);
    let packed_up = packed_zeros(n_experts, n_ff, n_embd);
    let packed_down = packed_zeros(n_experts, n_embd, n_ff);

    // gate_inp = large positive → silu_div ≈ 1.0.
    let shexp_gate_inp = Tensor::from_vec(vec![100.0f32; n_embd], (1, n_embd), &dev()).unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true,
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let xs = Tensor::from_vec(x_data.clone(), (1, 1, n_embd), &dev()).unwrap();

    let out = block.forward(&xs, ForwardMode::Prefill).unwrap();
    let out_vec: Vec<f32> = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    let expected = silu_mul_ref(&x_data);

    for (i, (got, exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "element {i}: got {got}, expected {exp} (1.0 * silu(x)*x)"
        );
    }
}

/// Shared expert gate = 0 (gate_inp weight = large negative → silu_div → 0.0).
/// Routed experts = identity SwiGLU, shared = identity SwiGLU.
/// Expected: output = silu(x) * x (routed only, shared contributes 0).
#[test]
fn test_shared_gate_zero() {
    let n_embd = 4usize;
    let n_ff = 4usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize;

    let router_w = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    let packed_gate = packed_identity(n_experts, n_embd);
    let packed_up = packed_identity(n_experts, n_embd);
    let packed_down = packed_identity(n_experts, n_embd);

    let shexp_gate_inp = Tensor::from_vec(vec![-100.0f32; n_embd], (1, n_embd), &dev()).unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true,
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let xs = Tensor::from_vec(x_data.clone(), (1, 1, n_embd), &dev()).unwrap();

    let out = block.forward(&xs, ForwardMode::Prefill).unwrap();
    let out_vec: Vec<f32> = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    let expected = silu_mul_ref(&x_data);

    for (i, (got, exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "element {i}: got {got}, expected {exp} (silu(x)*x, shared gate=0)"
        );
    }
}

/// Composition: routed = identity (weight 0.5 after norm), shared = identity (gate 0.5).
/// Expected: 0.5 * silu(x)*x + 0.5 * silu(x)*x = silu(x)*x.
#[test]
fn test_composition_routed_plus_shared() {
    let n_embd = 4usize;
    let n_ff = 4usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 2usize; // top-2 → both experts selected

    // Router: uniform weights → softmax uniform → both experts selected with equal prob.
    // After norm_topk_prob: each weight = 0.5.
    let router_w = Tensor::from_vec(
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    // Both experts are identity.
    let packed_gate = packed_identity(n_experts, n_embd);
    let packed_up = packed_identity(n_experts, n_embd);
    let packed_down = packed_identity(n_experts, n_embd);

    // Shared: gate_inp = 0 → gate_scalar = 0.5.
    let shexp_gate_inp = Tensor::from_vec(vec![0.0f32; n_embd], (1, n_embd), &dev()).unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true,
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let xs = Tensor::from_vec(x_data.clone(), (1, 1, n_embd), &dev()).unwrap();

    let out = block.forward(&xs, ForwardMode::Prefill).unwrap();
    let out_vec: Vec<f32> = out.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();

    // Routed: 2 experts × 0.5 weight × silu(x)*x = 0.5*silu(x)*x + 0.5*silu(x)*x = silu(x)*x
    // Shared: 0.5 * silu(x)*x
    // Total: silu(x)*x + 0.5*silu(x)*x = 1.5 * silu(x)*x
    let expected: Vec<f32> = silu_mul_ref(&x_data).iter().map(|v| v * 1.5).collect();

    for (i, (got, exp)) in out_vec.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < TOL,
            "element {i}: got {got}, expected {exp} (1.5 * silu(x)*x)"
        );
    }
}

/// Router determinism: known router weights select expected experts.
#[test]
fn test_router_topk_selection() {
    let n_embd = 4usize;
    let n_experts = 4usize;
    let n_experts_per_tok = 2usize;

    // Router: each expert responds to one dimension.
    // expert 0 = [1,0,0,0], expert 1 = [0,1,0,0], expert 2 = [0,0,1,0], expert 3 = [0,0,0,1]
    let router_w = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 0.0, // expert 0
            0.0, 1.0, 0.0, 0.0, // expert 1
            0.0, 0.0, 1.0, 0.0, // expert 2
            0.0, 0.0, 0.0, 1.0, // expert 3
        ],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    let router = MoeRouter::new(
        Linear::new(router_w, None),
        n_experts,
        n_experts_per_tok,
        true, // norm_topk_prob
    );

    // Input: x = [4, 3, 2, 1] → logits = [4, 3, 2, 1]
    // top-2: experts 0 and 1 (highest logits).
    let xs = Tensor::from_vec(vec![4.0f32, 3.0, 2.0, 1.0], (1, n_embd), &dev()).unwrap();
    let route = router.route_topk(&xs).unwrap();

    assert_eq!(route.experts.len(), 1, "one token");
    assert_eq!(route.experts[0].len(), 2, "top-2");
    assert_eq!(route.experts[0][0], 0, "first selected expert = 0 (logit 4)");
    assert_eq!(route.experts[0][1], 1, "second selected expert = 1 (logit 3)");

    // norm_topk_prob: weights sum to 1.
    let sum: f32 = route.weights[0].iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "normalized weights sum to 1, got {sum}");
}

/// Router tie-break: equal logits → lower expert index wins.
#[test]
fn test_router_tie_break_lower_index() {
    let n_embd = 2usize;
    let n_experts = 4usize;
    let n_experts_per_tok = 2usize;

    // All experts have identical weights → all logits equal → tie.
    let router_w = Tensor::from_vec(vec![1.0f32; n_experts * n_embd], (n_experts, n_embd), &dev())
        .unwrap();

    let router = MoeRouter::new(
        Linear::new(router_w, None),
        n_experts,
        n_experts_per_tok,
        true,
    );

    let xs = Tensor::from_vec(vec![1.0f32, 1.0], (1, n_embd), &dev()).unwrap();
    let route = router.route_topk(&xs).unwrap();

    // All logits equal → strict `>` tie-break → experts 0 and 1 selected.
    assert_eq!(route.experts[0][0], 0, "tie-break: expert 0 first");
    assert_eq!(route.experts[0][1], 1, "tie-break: expert 1 second");
}

/// Chunked vs unchunked: forward with Prefill vs DecodeBatch must be identical
/// for the reference backend (math is the same, only mode label differs).
#[test]
fn test_chunked_vs_unchunked_tolerance() {
    let n_embd = 4usize;
    let n_ff = 4usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize;

    let router_w = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    let packed_gate = packed_identity(n_experts, n_embd);
    let packed_up = packed_identity(n_experts, n_embd);
    let packed_down = packed_identity(n_experts, n_embd);

    let shexp_gate_inp = Tensor::from_vec(vec![0.0f32; n_embd], (1, n_embd), &dev()).unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true,
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
    let xs = Tensor::from_vec(x_data, (1, 2, n_embd), &dev()).unwrap();

    let out_prefill = block
        .forward(&xs, ForwardMode::Prefill)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let out_decode = block
        .forward(&xs, ForwardMode::DecodeBatch)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    for (i, (a, b)) in out_prefill.iter().zip(out_decode.iter()).enumerate() {
        assert!(
            (a - b).abs() < TOL,
            "chunked vs unchunked mismatch at {i}: prefill={a}, decode={b}"
        );
    }
}

/// Output shape preservation: [batch, seq_len, n_embd] in → same out.
#[test]
fn test_output_shape() {
    let n_embd = 4usize;
    let n_ff = 4usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize;

    let router_w = Tensor::from_vec(vec![1.0f32; n_experts * n_embd], (n_experts, n_embd), &dev())
        .unwrap();
    let packed_gate = packed_identity(n_experts, n_embd);
    let packed_up = packed_identity(n_experts, n_embd);
    let packed_down = packed_identity(n_experts, n_embd);
    let shexp_gate_inp = Tensor::from_vec(vec![0.0f32; n_embd], (1, n_embd), &dev()).unwrap();
    let shexp_gate = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_up = Tensor::from_vec(identity_flat(n_ff), (n_ff, n_embd), &dev()).unwrap();
    let shexp_down = Tensor::from_vec(identity_flat(n_embd), (n_embd, n_ff), &dev()).unwrap();

    let block = build_moe_block(
        n_embd,
        n_ff,
        n_experts,
        n_experts_per_tok,
        true,
        &router_w,
        &packed_gate,
        &packed_up,
        &packed_down,
        &shexp_gate_inp,
        &shexp_gate,
        &shexp_up,
        &shexp_down,
    );

    let xs = Tensor::zeros((2, 3, n_embd), DType::F32, &dev()).unwrap();
    let out = block.forward(&xs, ForwardMode::Prefill).unwrap();
    assert_eq!(out.dims(), [2, 3, n_embd]);
}

/// norm_topk_prob=false: weights are raw softmax probs (not normalized).
#[test]
fn test_no_norm_topk_prob() {
    let n_embd = 2usize;
    let n_experts = 2usize;
    let n_experts_per_tok = 1usize;

    // Router: expert 0 = [10, 0], expert 1 = [0, 1].
    // Input x = [1, 0] → logits = [10, 0] → softmax ≈ [1.0, ~0].
    let router_w = Tensor::from_vec(
        vec![10.0f32, 0.0, 0.0, 1.0],
        (n_experts, n_embd),
        &dev(),
    )
    .unwrap();

    let router = MoeRouter::new(
        Linear::new(router_w, None),
        n_experts,
        n_experts_per_tok,
        false, // norm_topk_prob = false
    );

    let xs = Tensor::from_vec(vec![1.0f32, 0.0], (1, n_embd), &dev()).unwrap();
    let route = router.route_topk(&xs).unwrap();

    // With norm=false, weight = raw softmax prob (close to 1.0 but not exactly 1.0).
    let w = route.weights[0][0];
    assert!(w > 0.99, "raw softmax prob should be ~1.0, got {w}");
    assert!(w < 1.0, "raw softmax prob < 1.0 (not normalized), got {w}");
}

/// expert_slice helper: verify narrow/squeeze produces correct 2D slice.
#[test]
fn test_expert_slice_helper() {
    let n_experts = 3;
    let n = 4;
    let packed = packed_identity(n_experts, n);

    let e0 = expert_slice(&packed, 0, n, n);
    let e0_vec: Vec<f32> = e0.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(e0_vec, identity_flat(n));

    let e1 = expert_slice(&packed, 1, n, n);
    let e1_vec: Vec<f32> = e1.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(e1_vec, identity_flat(n));
}