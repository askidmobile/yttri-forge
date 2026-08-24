mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Id {
    Affine,
    Binary,
    Cast,
    Conv,
    DeltaRule,
    /// Phase 2: true batched decode (сть slot B) — Qwen35-batching.
    DeltaRuleBatched,
    /// Split-K flash-decoding (длинный KV, seqlen_q=1).
    FlashDecode,
    Fill,
    Indexing,
    Moe,
    /// Phase 3: device-only MoE routing + quantized expert kernels (PTX).
    MoeRouter,
    MoeQuantized,
    /// Tensor-Core MMA MMQ (llama.cpp mul_mat_q) для плотного prefill — Qwen35-batching.
    MmqDense,
    Quantized,
    Reduce,
    Sort,
    Ternary,
    Unary,
}

pub const ALL_IDS: [Id; 18] = [
    Id::Affine,
    Id::Binary,
    Id::Cast,
    Id::Conv,
    Id::DeltaRule,
    Id::DeltaRuleBatched,
    Id::FlashDecode,
    Id::Fill,
    Id::Indexing,
    Id::Moe,
    Id::MoeRouter,
    Id::MoeQuantized,
    Id::MmqDense,
    Id::Quantized,
    Id::Reduce,
    Id::Sort,
    Id::Ternary,
    Id::Unary,
];

pub struct Module {
    index: usize,
    ptx: &'static str,
}

impl Module {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

const fn module_index(id: Id) -> usize {
    let mut i = 0;
    while i < ALL_IDS.len() {
        if ALL_IDS[i] as u32 == id as u32 {
            return i;
        }
        i += 1;
    }
    panic!("id not found")
}

macro_rules! mdl {
    ($cst:ident, $id:ident) => {
        pub const $cst: Module = Module {
            index: module_index(Id::$id),
            ptx: ptx::$cst,
        };
    };
}

mdl!(AFFINE, Affine);
mdl!(BINARY, Binary);
mdl!(CAST, Cast);
mdl!(CONV, Conv);
mdl!(DELTA_RULE, DeltaRule);
mdl!(FLASH_DECODE, FlashDecode);
mdl!(DELTA_RULE_BATCHED, DeltaRuleBatched);
mdl!(FILL, Fill);
mdl!(INDEXING, Indexing);
mdl!(MOE, Moe);
mdl!(MOE_ROUTER, MoeRouter);
mdl!(MOE_QUANTIZED, MoeQuantized);
mdl!(CANDLE_MMQ_DENSE, MmqDense);
mdl!(QUANTIZED, Quantized);
mdl!(REDUCE, Reduce);
mdl!(SORT, Sort);
mdl!(TERNARY, Ternary);
mdl!(UNARY, Unary);

pub mod ffi;
