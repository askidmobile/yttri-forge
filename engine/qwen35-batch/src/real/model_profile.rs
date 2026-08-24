//! Model profile and fail-fast GGUF validation (Phase 1).
//!
//! Produces an immutable `ModelProfile` from a `gguf_file::Content` before
//! heavy tensor loading. Validates architecture, block layout, expert config,
//! and tensor contracts for both dense `qwen35` and MoE `qwen35moe`.
//!
//! Errors are aggregated where safe so startup reports all incompatible
//! properties at once rather than failing one-at-a-time.

use candle_core::quantized::gguf_file::{Content, TensorInfo, Value};
use candle_core::quantized::GgmlDType;
use candle_core::Result;
use std::collections::{BTreeMap, HashSet};

/// Clone a TensorInfo manually since gguf_file::TensorInfo does not impl Clone.
fn clone_tensor_info(ti: &TensorInfo) -> TensorInfo {
    TensorInfo {
        ggml_dtype: ti.ggml_dtype,
        shape: ti.shape.clone(),
        offset: ti.offset,
    }
}

/// Runtime architecture selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Dense Qwen3.5/Qwen3.6 hybrid DeltaNet/attention trunk (`general.architecture=qwen35`).
    DenseQwen35,
    /// MoE Qwen3.6 35B-A3B (`general.architecture=qwen35moe`).
    Qwen35Moe,
}

impl Architecture {
    /// Parse from the `general.architecture` GGUF metadata value.
    pub fn from_metadata(md: &BTreeMap<String, Value>) -> Result<Self> {
        let arch = md
            .get("general.architecture")
            .ok_or_else(|| candle_core::Error::Msg("missing general.architecture".into()))?;
        let s = match arch {
            Value::String(s) => s.as_str(),
            _ => return Err(candle_core::Error::Msg("general.architecture not a string".into())),
        };
        match s {
            "qwen35" => Ok(Self::DenseQwen35),
            "qwen35moe" => Ok(Self::Qwen35Moe),
            other => Err(candle_core::Error::Msg(format!(
                "unsupported architecture: '{other}' — supported: qwen35, qwen35moe"
            ))),
        }
    }

    pub fn metadata_prefix(&self) -> &'static str {
        match self {
            Self::DenseQwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
        }
    }
}

/// Stable fingerprint derived from architecture metadata + tensor manifest.
/// Does NOT include the local file path — reproducible across machines.
#[derive(Debug, Clone)]
pub struct ModelFingerprint {
    pub hash: String,
}

impl ModelFingerprint {
    fn compute(
        arch: Architecture,
        metadata: &BTreeMap<String, Value>,
        tensor_manifest: &[(String, GgmlDType, Vec<usize>)],
        file_size: u64,
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h = DefaultHasher::new();
        arch.metadata_prefix().hash(&mut h);
        // Architecture-relevant metadata keys (sorted by BTreeMap iteration).
        let prefix = arch.metadata_prefix();
        let arch_keys = [
            "general.architecture",
            "general.file_type",
            "general.name",
            "tokenizer.ggml.model",
        ];
        for key in arch_keys {
            if let Some(v) = metadata.get(key) {
                key.hash(&mut h);
                format!("{v:?}").hash(&mut h);
            }
        }
        // Prefix-specific metadata keys.
        let prefix_keys = [
            format!("{prefix}.block_count"),
            format!("{prefix}.context_length"),
            format!("{prefix}.embedding_length"),
            format!("{prefix}.feed_forward_length"),
            format!("{prefix}.attention.head_count"),
            format!("{prefix}.attention.head_count_kv"),
            format!("{prefix}.attention.key_length"),
            format!("{prefix}.attention.value_length"),
        ];
        for key in prefix_keys {
            if let Some(v) = metadata.get(&key) {
                key.hash(&mut h);
                format!("{v:?}").hash(&mut h);
            }
        }
        // MoE-specific keys.
        if matches!(arch, Architecture::Qwen35Moe) {
            let moe_keys = [
                "qwen35moe.expert_count",
                "qwen35moe.expert_used_count",
                "qwen35moe.shared_expert_count",
            ];
            for key in moe_keys {
                if let Some(v) = metadata.get(key) {
                    key.hash(&mut h);
                    format!("{v:?}").hash(&mut h);
                }
            }
        }
        // Tensor manifest.
        for (name, dtype, shape) in tensor_manifest {
            name.hash(&mut h);
            format!("{dtype:?}").hash(&mut h);
            shape.hash(&mut h);
        }
        file_size.hash(&mut h);
        Self {
            hash: format!("qwen35-profile-{:016x}", h.finish()),
        }
    }
}

/// Immutable model profile produced before heavy tensor loading.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub architecture: Architecture,
    pub block_count: usize,
    pub hidden_size: usize,
    pub context_length: usize,
    pub full_attention_interval: usize,
    pub feed_forward_length: usize,
    pub attention_head_count: usize,
    pub attention_head_count_kv: usize,
    pub attention_key_length: usize,
    pub attention_value_length: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
    /// MoE-specific fields (None for dense).
    pub routed_experts: Option<usize>,
    pub experts_per_token: Option<usize>,
    pub routed_intermediate: Option<usize>,
    pub shared_intermediate: Option<usize>,
    pub shared_expert_count: Option<usize>,
    pub router_norm_topk: Option<bool>,
    /// Distinct quant dtypes present in the GGUF tensor infos.
    pub quant_set: HashSet<GgmlDType>,
    pub fingerprint: ModelFingerprint,
}

/// Aggregated validation errors — all problems reported at once.
#[derive(Debug, Default)]
pub struct ValidationErrors {
    pub errors: Vec<String>,
}

impl ValidationErrors {
    fn push(&mut self, msg: String) {
        self.errors.push(msg);
    }
    fn push_missing(&mut self, key: &str) {
        self.errors.push(format!("missing metadata key: {key}"));
    }
    fn push_tensor_missing(&mut self, name: &str) {
        self.errors.push(format!("missing required tensor: {name}"));
    }
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
    pub fn into_result(self) -> Result<()> {
        if self.errors.is_empty() {
            Ok(())
        } else {
            let joined = self.errors.join("; ");
            Err(candle_core::Error::Msg(format!(
                "GGUF validation failed ({} error(s)): {joined}",
                self.errors.len()
            )))
        }
    }
}

/// Helper to read a u32 metadata value.
fn md_u32(md: &BTreeMap<String, Value>, key: &str, errs: &mut ValidationErrors) -> Option<usize> {
    match md.get(key) {
        Some(v) => match v.to_u32() {
            Ok(n) => Some(n as usize),
            Err(_) => {
                errs.push(format!("metadata key {key} is not a u32"));
                None
            }
        },
        None => {
            errs.push_missing(key);
            None
        }
    }
}

/// Helper to read an f32 metadata value (optional with default).
fn md_f32_opt(md: &BTreeMap<String, Value>, key: &str, default: f32) -> f32 {
    md.get(key)
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(default)
}

/// Helper to read an f32 metadata value (required).
fn md_f32_req(md: &BTreeMap<String, Value>, key: &str, errs: &mut ValidationErrors) -> Option<f32> {
    match md.get(key) {
        Some(v) => match v.to_f32() {
            Ok(n) => Some(n),
            Err(_) => {
                errs.push(format!("metadata key {key} is not an f32"));
                None
            }
        },
        None => {
            errs.push_missing(key);
            None
        }
    }
}

/// Helper to read a bool metadata value (optional).
fn md_bool_opt(md: &BTreeMap<String, Value>, key: &str) -> Option<bool> {
    md.get(key).and_then(|v| v.to_bool().ok())
}

/// Check that a required tensor exists in the GGUF tensor_infos.
fn require_tensor(
    tensor_infos: &BTreeMap<String, TensorInfo>,
    name: &str,
    errs: &mut ValidationErrors,
) {
    if !tensor_infos.contains_key(name) {
        errs.push_tensor_missing(name);
    }
}

fn require_tensor_contract(
    tensor_infos: &BTreeMap<String, TensorInfo>,
    name: &str,
    shape: &[usize],
    dtypes: &[GgmlDType],
    errs: &mut ValidationErrors,
) {
    match tensor_infos.get(name) {
        None => errs.push_tensor_missing(name),
        Some(info) => {
            if info.shape.dims() != shape {
                errs.push(format!(
                    "tensor {name} shape {:?}, expected {shape:?}",
                    info.shape.dims()
                ));
            }
            if !dtypes.contains(&info.ggml_dtype) {
                errs.push(format!(
                    "tensor {name} dtype {:?}, expected one of {dtypes:?}",
                    info.ggml_dtype
                ));
            }
        }
    }
}

/// Immutable Qwen3.5 Vision component contract.
#[derive(Debug, Clone)]
pub struct VisionProfile {
    pub block_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_count: usize,
    pub projection_dim: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub position_embeddings: usize,
    pub quant_set: HashSet<GgmlDType>,
}

impl VisionProfile {
    pub fn read_and_validate(ct: &Content) -> Result<Self> {
        Self::read_and_validate_with_policy(ct, true)
    }

    pub fn read_reference(ct: &Content) -> Result<Self> {
        Self::read_and_validate_with_policy(ct, false)
    }

    fn read_and_validate_with_policy(ct: &Content, production_q8: bool) -> Result<Self> {
        let mut errs = ValidationErrors::default();
        let metadata: BTreeMap<String, Value> = ct.metadata.clone().into_iter().collect();
        let tensors: BTreeMap<String, TensorInfo> = ct
            .tensor_infos
            .iter()
            .map(|(name, info)| (name.clone(), clone_tensor_info(info)))
            .collect();

        match metadata.get("general.architecture") {
            Some(Value::String(value)) if value == "clip" => {}
            Some(value) => errs.push(format!(
                "general.architecture={value:?}, expected clip for Qwen3.5 Vision"
            )),
            None => errs.push_missing("general.architecture"),
    }
        let block_count = md_u32(&metadata, "clip.vision.block_count", &mut errs).unwrap_or(0);
        let hidden_size = md_u32(&metadata, "clip.vision.embedding_length", &mut errs).unwrap_or(0);
        let intermediate_size =
            md_u32(&metadata, "clip.vision.feed_forward_length", &mut errs).unwrap_or(0);
        let head_count =
            md_u32(&metadata, "clip.vision.attention.head_count", &mut errs).unwrap_or(0);
        let projection_dim =
            md_u32(&metadata, "clip.vision.projection_dim", &mut errs).unwrap_or(0);
        let patch_size = md_u32(&metadata, "clip.vision.patch_size", &mut errs).unwrap_or(0);
        let spatial_merge_size =
            md_u32(&metadata, "clip.vision.spatial_merge_size", &mut errs).unwrap_or(0);
        let image_size = md_u32(&metadata, "clip.vision.image_size", &mut errs).unwrap_or(0);
        let position_embeddings = (image_size / patch_size.max(1)).pow(2);

        for (name, actual, expected) in [
            ("block_count", block_count, 24),
            ("hidden_size", hidden_size, 1024),
            ("intermediate_size", intermediate_size, 4096),
            ("head_count", head_count, 16),
            ("projection_dim", projection_dim, 2560),
            ("patch_size", patch_size, 16),
            ("spatial_merge_size", spatial_merge_size, 2),
            ("position_embeddings", position_embeddings, 2304),
        ] {
            if actual != expected {
                errs.push(format!("Vision {name}={actual}, expected {expected}"));
}
        }
        if tensors.len() != 298 {
            errs.push(format!(
                "Vision tensor count={}, expected 298",
                tensors.len()
            ));
        }

        let f32_only = [GgmlDType::F32];
        let matrix = [
            GgmlDType::Q8_0,
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
        ];
        let sensitive_matrix = if production_q8 {
            &[GgmlDType::BF16, GgmlDType::F32][..]
        } else {
            &matrix[..]
        };
        require_tensor_contract(
            &tensors,
            "v.patch_embd.weight",
            &[1024, 3, 16, 16],
            &matrix,
            &mut errs,
        );
        require_tensor_contract(
            &tensors,
            "v.patch_embd.weight.1",
            &[1024, 3, 16, 16],
            &matrix,
            &mut errs,
        );
        require_tensor_contract(&tensors, "v.patch_embd.bias", &[1024], &f32_only, &mut errs);
        require_tensor_contract(
            &tensors,
            "v.position_embd.weight",
            &[2304, 1024],
            &matrix,
            &mut errs,
        );
        require_tensor_contract(&tensors, "v.post_ln.weight", &[1024], &f32_only, &mut errs);
        require_tensor_contract(&tensors, "v.post_ln.bias", &[1024], &f32_only, &mut errs);
        require_tensor_contract(
            &tensors,
            "mm.0.weight",
            &[4096, 4096],
            sensitive_matrix,
            &mut errs,
        );
        require_tensor_contract(&tensors, "mm.0.bias", &[4096], &f32_only, &mut errs);
        require_tensor_contract(
            &tensors,
            "mm.2.weight",
            &[2560, 4096],
            sensitive_matrix,
            &mut errs,
        );
        require_tensor_contract(&tensors, "mm.2.bias", &[2560], &f32_only, &mut errs);
        for layer in 0..block_count {
            let prefix = format!("v.blk.{layer}");
            for (suffix, shape) in [
                ("attn_qkv.weight", vec![3072, 1024]),
                ("attn_out.weight", vec![1024, 1024]),
                ("ffn_up.weight", vec![4096, 1024]),
                ("ffn_down.weight", vec![1024, 4096]),
            ] {
                let dtypes = if production_q8
                    && suffix == "ffn_down.weight"
                    && (1..20).contains(&layer)
                {
                    &[GgmlDType::Q8_0][..]
                } else {
                    sensitive_matrix
                };
                require_tensor_contract(
                    &tensors,
                    &format!("{prefix}.{suffix}"),
                    &shape,
                    dtypes,
                    &mut errs,
                );
            }
            for (suffix, size) in [
                ("attn_qkv.bias", 3072),
                ("attn_out.bias", 1024),
                ("ffn_up.bias", 4096),
                ("ffn_down.bias", 1024),
                ("ln1.weight", 1024),
                ("ln1.bias", 1024),
                ("ln2.weight", 1024),
                ("ln2.bias", 1024),
            ] {
                require_tensor_contract(
                    &tensors,
                    &format!("{prefix}.{suffix}"),
                    &[size],
                    &f32_only,
                    &mut errs,
                );
            }
        }
        let quant_set = tensors.values().map(|info| info.ggml_dtype).collect();
        errs.into_result()?;
        Ok(Self {
            block_count,
            hidden_size,
            intermediate_size,
            head_count,
            projection_dim,
            patch_size,
            spatial_merge_size,
            position_embeddings,
            quant_set,
        })
    }
}

/// Thin Qwen3.5 MTP component contract. Shared embedding/output remain in Text.
#[derive(Debug, Clone)]
pub struct MtpProfile {
    /// Индекс MTP-блока (blk.{mtp_block}): у тонкого 4B-файла 32, у Qwen3.8
    /// (MTP встроен в основной GGUF) — block_count-1 = 64.
    pub mtp_block: usize,
    pub hidden_size: usize,
    pub context_length: usize,
    pub intermediate_size: usize,
    pub head_count: usize,
    pub kv_head_count: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub quant_set: HashSet<GgmlDType>,
}

impl MtpProfile {
    pub fn read_and_validate(ct: &Content, text: &ModelProfile) -> Result<Self> {
        let mut errs = ValidationErrors::default();
        let metadata: BTreeMap<String, Value> = ct.metadata.clone().into_iter().collect();
        let tensors: BTreeMap<String, TensorInfo> = ct
            .tensor_infos
            .iter()
            .map(|(name, info)| (name.clone(), clone_tensor_info(info)))
            .collect();
        match Architecture::from_metadata(&metadata) {
            Ok(Architecture::DenseQwen35) => {}
            Ok(other) => errs.push(format!("MTP architecture {other:?}, expected dense qwen35")),
            Err(error) => errs.push(format!("{error}")),
        }
        let blocks = md_u32(&metadata, "qwen35.block_count", &mut errs).unwrap_or(0);
        let nextn = md_u32(&metadata, "qwen35.nextn_predict_layers", &mut errs).unwrap_or(0);
        let hidden = md_u32(&metadata, "qwen35.embedding_length", &mut errs).unwrap_or(0);
        let context = md_u32(&metadata, "qwen35.context_length", &mut errs).unwrap_or(0);
        let intermediate =
            md_u32(&metadata, "qwen35.feed_forward_length", &mut errs).unwrap_or(0);
        let heads = md_u32(&metadata, "qwen35.attention.head_count", &mut errs).unwrap_or(0);
        let kv_heads =
            md_u32(&metadata, "qwen35.attention.head_count_kv", &mut errs).unwrap_or(0);
        let key_length =
            md_u32(&metadata, "qwen35.attention.key_length", &mut errs).unwrap_or(0);
        let value_length =
            md_u32(&metadata, "qwen35.attention.value_length", &mut errs).unwrap_or(0);
        let rms = md_f32_req(
            &metadata,
            "qwen35.attention.layer_norm_rms_epsilon",
            &mut errs,
        )
        .unwrap_or(0.0) as f64;
        // Контракт параметрический: 4B-MTP - тонкий отдельный GGUF (33 блока,
        // hidden 2560), Qwen3.8+ - MTP встроен в основной файл (blk.<last>).
        // Всё выводится из метаданных + text-профиля, кроме инвариантов
        // mtp.rs: head_dim=256, rope_dim=64, rope_base=1e7 (константы форварда).
        if nextn != 1 {
            errs.push(format!("MTP nextn_predict_layers={nextn}, expected 1"));
        }
        if key_length != 256 || value_length != 256 {
            errs.push(format!(
                "MTP head_dim {key_length}/{value_length}, mtp.rs supports only 256"
            ));
        }
        if text.architecture != Architecture::DenseQwen35
            || text.hidden_size != hidden
            || text.context_length != context
            || text.feed_forward_length != intermediate
            || text.attention_head_count != heads
            || text.attention_head_count_kv != kv_heads
            || text.attention_key_length != key_length
            || text.attention_value_length != value_length
        {
            errs.push("MTP component is incompatible with shared Text profile".into());
        }
        // Тонкий файл: text-транк block_count = blocks - nextn (33-1=32).
        // Встроенный: text-профиль уже вычел nextn (см. model_profile block_count).
        if text.block_count + nextn != blocks && text.block_count != blocks {
            errs.push(format!(
                "MTP block_count={blocks} inconsistent with text trunk {}",
                text.block_count
            ));
        }
        let mtp_block = blocks - nextn;
        let prefix = format!("blk.{mtp_block}");
        let head_dim = key_length;
        // Кванты матриц: 4B-MTP был Q8_0; у Qwen3.8 IQ4_XS/Q3_K. Разрешаем
        // любой поддерживаемый CUDA-путём квант (гейт численности - teacher-forced).
        let matrix = [
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q4_0,
            GgmlDType::IQ2XXS,
            GgmlDType::IQ2XS,
            GgmlDType::IQ2S,
            GgmlDType::IQ3XXS,
            GgmlDType::IQ3S,
            GgmlDType::IQ4XS,
            GgmlDType::IQ1M,
            GgmlDType::F16,
        ];
        let norm = [GgmlDType::F32];
        for (name, shape) in [
            (format!("{prefix}.attn_q.weight"), vec![heads * head_dim * 2, hidden]),
            (format!("{prefix}.attn_k.weight"), vec![kv_heads * head_dim, hidden]),
            (format!("{prefix}.attn_v.weight"), vec![kv_heads * head_dim, hidden]),
            (format!("{prefix}.attn_output.weight"), vec![hidden, heads * head_dim]),
            (format!("{prefix}.ffn_gate.weight"), vec![intermediate, hidden]),
            (format!("{prefix}.ffn_up.weight"), vec![intermediate, hidden]),
            (format!("{prefix}.ffn_down.weight"), vec![hidden, intermediate]),
            (format!("{prefix}.nextn.eh_proj.weight"), vec![hidden, 2 * hidden]),
        ] {
            require_tensor_contract(&tensors, &name, &shape, &matrix, &mut errs);
        }
        for (name, shape) in [
            (format!("{prefix}.attn_norm.weight"), vec![hidden]),
            (format!("{prefix}.post_attention_norm.weight"), vec![hidden]),
            (format!("{prefix}.attn_q_norm.weight"), vec![head_dim]),
            (format!("{prefix}.attn_k_norm.weight"), vec![head_dim]),
            (format!("{prefix}.nextn.enorm.weight"), vec![hidden]),
            (format!("{prefix}.nextn.hnorm.weight"), vec![hidden]),
            (format!("{prefix}.nextn.shared_head_norm.weight"), vec![hidden]),
        ] {
            require_tensor_contract(&tensors, &name, &shape, &norm, &mut errs);
        }
        // Для тонкого файла (только MTP-тензоры) запрещаем лишнее; встроенный
        // основной GGUF содержит транк - проверка не применима.
        let mtp_only: Vec<_> = tensors
            .keys()
            .filter(|name| name.starts_with(&format!("{prefix}.")))
            .collect();
        if tensors.len() == mtp_only.len() && tensors.len() != 15 {
            errs.push(format!(
                "thin MTP tensor count={}, expected 15",
                tensors.len()
            ));
        }
        let quant_set = tensors
            .iter()
            .filter(|(name, _)| name.starts_with(&format!("{prefix}.")))
            .map(|(_, info)| info.ggml_dtype)
            .collect();
        errs.into_result()?;
        Ok(Self {
            mtp_block,
            hidden_size: hidden,
            context_length: context,
            intermediate_size: intermediate,
            head_count: heads,
            kv_head_count: kv_heads,
            head_dim: key_length,
            rms_norm_eps: rms,
            quant_set,
        })
    }
}

/// Check tensors for a single block at the given layer index.
/// GGUF tensor names have NO architecture prefix (`blk.N....`); the prefix is
/// only for metadata keys. Hybrid trunk: DeltaNet blocks (linear recurrence)
/// interleave with full-attention blocks every `full_attention_interval`.
fn require_block_tensors(
    tensor_infos: &BTreeMap<String, TensorInfo>,
    arch: Architecture,
    layer: usize,
    full_attention_interval: usize,
    errs: &mut ValidationErrors,
) {
    let prefix = format!("blk.{layer}");
    // Common norms for both block types.
    require_tensor(tensor_infos, &format!("{prefix}.attn_norm.weight"), errs);
    require_tensor(
        tensor_infos,
        &format!("{prefix}.post_attention_norm.weight"),
        errs,
    );
    // Sequence mixer depends on hybrid layout.
    let interval = full_attention_interval.max(1);
    if (layer + 1) % interval == 0 {
        // Full attention block.
        for t in [
            format!("{prefix}.attn_q.weight"),
            format!("{prefix}.attn_k.weight"),
            format!("{prefix}.attn_v.weight"),
            format!("{prefix}.attn_output.weight"),
        ] {
            require_tensor(tensor_infos, &t, errs);
        }
    } else {
        // DeltaNet block.
        for t in [
            format!("{prefix}.attn_qkv.weight"),
            format!("{prefix}.attn_gate.weight"),
            format!("{prefix}.ssm_beta.weight"),
            format!("{prefix}.ssm_alpha.weight"),
            format!("{prefix}.ssm_out.weight"),
            format!("{prefix}.ssm_dt.bias"),
            format!("{prefix}.ssm_a"),
            format!("{prefix}.ssm_conv1d.weight"),
            format!("{prefix}.ssm_norm.weight"),
        ] {
            require_tensor(tensor_infos, &t, errs);
        }
    }
    // Feed-forward tensors depend on architecture.
    match arch {
        Architecture::DenseQwen35 => {
            let ffn = [
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.ffn_up.weight"),
                format!("{prefix}.ffn_down.weight"),
            ];
            for t in &ffn {
                require_tensor(tensor_infos, t, errs);
            }
        }
        Architecture::Qwen35Moe => {
            // MoE: router + shared expert + packed routed experts.
            let moe_tensors = [
                format!("{prefix}.ffn_gate_inp.weight"),  // router
                format!("{prefix}.ffn_gate_exps.weight"), // routed gate
                format!("{prefix}.ffn_up_exps.weight"),   // routed up
                format!("{prefix}.ffn_down_exps.weight"), // routed down
                format!("{prefix}.ffn_gate_shexp.weight"), // shared gate
                format!("{prefix}.ffn_up_shexp.weight"),  // shared up
                format!("{prefix}.ffn_down_shexp.weight"), // shared down
            ];
            for t in &moe_tensors {
                require_tensor(tensor_infos, t, errs);
            }
        }
    }
}

impl ModelProfile {
    /// Read and validate a GGUF `Content` before heavy tensor loading.
    ///
    /// Returns a `ModelProfile` on success, or an aggregated error listing
    /// all validation failures.
    pub fn read_and_validate(ct: &Content, file_size: u64) -> Result<Self> {
        let mut errs = ValidationErrors::default();

        // Sorted metadata and tensor_infos for deterministic iteration.
        let metadata: BTreeMap<String, Value> = ct.metadata.clone().into_iter().collect();
        let tensor_infos: BTreeMap<String, TensorInfo> = ct
            .tensor_infos
            .iter()
            .map(|(k, v)| (k.clone(), clone_tensor_info(v)))
            .collect();

        // 1. Architecture.
        let architecture = match Architecture::from_metadata(&metadata) {
            Ok(a) => a,
            Err(e) => {
                errs.push(format!("{e}"));
                // Can't continue without architecture.
                return errs.into_result().map(|_| unreachable!());
            }
        };
        let prefix = architecture.metadata_prefix();

        // 2. Required metadata for both architectures.
        let block_count_raw = md_u32(&metadata, &format!("{prefix}.block_count"), &mut errs)
            .unwrap_or(0);
        // Qwen3.8+: block_count включает nextn/MTP-слои (blk.<last> с обычным
        // qkv-сплитом и nextn.*-тензорами) — валидируем только транк, MTP-слой
        // имеет свой контракт (mtp.rs). Симметрично build_model_common.
        let nextn_layers = metadata
            .get(&format!("{prefix}.nextn_predict_layers"))
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(0) as usize;
        let block_count = block_count_raw.saturating_sub(nextn_layers);
        let hidden_size = md_u32(&metadata, &format!("{prefix}.embedding_length"), &mut errs)
            .unwrap_or(0);
        let context_length = md_u32(&metadata, &format!("{prefix}.context_length"), &mut errs)
            .unwrap_or(0);
        // feed_forward_length: у dense обязателен; у qwen35moe отсутствует
        // (все FFN — MoE, unsloth-GGUF этот ключ не пишет).
        let feed_forward_length = if matches!(architecture, Architecture::Qwen35Moe) {
            metadata
                .get(&format!("{prefix}.feed_forward_length"))
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(0)
        } else {
            md_u32(&metadata, &format!("{prefix}.feed_forward_length"), &mut errs).unwrap_or(0)
        };
        let attention_head_count =
            md_u32(&metadata, &format!("{prefix}.attention.head_count"), &mut errs).unwrap_or(0);
        let attention_head_count_kv =
            md_u32(&metadata, &format!("{prefix}.attention.head_count_kv"), &mut errs)
                .unwrap_or(0);
        let attention_key_length =
            md_u32(&metadata, &format!("{prefix}.attention.key_length"), &mut errs).unwrap_or(0);
        let attention_value_length =
            md_u32(&metadata, &format!("{prefix}.attention.value_length"), &mut errs)
                .unwrap_or(0);
        let rms_norm_eps = md_f32_req(
            &metadata,
            &format!("{prefix}.attention.layer_norm_rms_epsilon"),
            &mut errs,
        )
        .map(|v| v as f64)
        .unwrap_or(0.0);
        let rope_freq_base = md_f32_opt(&metadata, &format!("{prefix}.rope.freq_base"), 10000.0);
        let full_attention_interval = md_u32(
            &metadata,
            &format!("{prefix}.full_attention_interval"),
            &mut errs,
        )
        .unwrap_or(4);

        // 3. MoE-specific metadata.
        let (routed_experts, experts_per_token, routed_intermediate, shared_intermediate, shared_expert_count, router_norm_topk) =
            match architecture {
                Architecture::Qwen35Moe => {
                    let re = md_u32(&metadata, "qwen35moe.expert_count", &mut errs);
                    let ept = md_u32(&metadata, "qwen35moe.expert_used_count", &mut errs);
                    // Ключи у разных GGUF-билдеров: llama.cpp-style,
                    // промежуточные, unsloth (expert_*_feed_forward_length).
                    let mut scratch = ValidationErrors::default();
                    let ri = [
                        "qwen35moe.feed_forward_length.experts",
                        "qwen35moe.intermediate_size_experts",
                        "qwen35moe.expert_feed_forward_length",
                    ]
                    .iter()
                    .find_map(|k| md_u32(&metadata, k, &mut scratch))
                    .or_else(|| {
                        errs.push_missing("qwen35moe.expert_feed_forward_length");
                        None
                    });
                    let si = [
                        "qwen35moe.feed_forward_length.shared_expert",
                        "qwen35moe.intermediate_size_shared_expert",
                        "qwen35moe.expert_shared_feed_forward_length",
                    ]
                    .iter()
                    .find_map(|k| md_u32(&metadata, k, &mut scratch))
                    .or_else(|| {
                        errs.push_missing("qwen35moe.expert_shared_feed_forward_length");
                        None
                    });
                    // shared_expert_count: в unsloth-GGUF отсутствует, default 1.
                    let sec = metadata
                        .get("qwen35moe.shared_expert_count")
                        .and_then(|v| v.to_u32().ok())
                        .map(|v| v as usize)
                        .or(Some(1));
                    let rnt = md_bool_opt(&metadata, "qwen35moe.router_norm_topk");
                    (re, ept, ri, si, sec, rnt)
                }
                Architecture::DenseQwen35 => (None, None, None, None, None, None),
            };

        // 4. Validate expert_used_count <= expert_count (if both present).
        if let (Some(re), Some(ept)) = (routed_experts, experts_per_token) {
            if ept > re {
                errs.push(format!(
                    "expert_used_count ({ept}) > expert_count ({re})"
                ));
            }
            // Target model: top-8.
            if ept != 8 {
                errs.push(format!(
                    "expert_used_count={ept}, expected 8 for Qwen3.6 35B-A3B"
                ));
            }
        }

        // 5. Validate block_count (target: 40 for Qwen3.6 35B-A3B MoE).
        if matches!(architecture, Architecture::Qwen35Moe) && block_count != 40 {
            errs.push(format!(
                "block_count={block_count}, expected 40 for Qwen3.6 35B-A3B"
            ));
        }

        // 6. Required global tensors.
        require_tensor(&tensor_infos, "token_embd.weight", &mut errs);
        require_tensor(&tensor_infos, "output_norm.weight", &mut errs);
        // output.weight may be tied — not strictly required.

        // 7. Per-block tensor validation (sample first + last block; full check
        //    for small models, sampled for large to keep validation fast).
        if block_count > 0 {
            let interval = full_attention_interval.max(1) as usize;
            // Always check block 0 and last block.
            for layer in [0usize, block_count - 1] {
                require_block_tensors(&tensor_infos, architecture, layer, interval, &mut errs);
            }
            // For MoE with <= 48 blocks, check all (40 is the target).
            if matches!(architecture, Architecture::Qwen35Moe) && block_count <= 48 {
                for layer in 0..block_count {
                    require_block_tensors(&tensor_infos, architecture, layer, interval, &mut errs);
                }
            }
        }

        // 8. Collect quant set from tensor infos.
        let quant_set: HashSet<GgmlDType> =
            tensor_infos.values().map(|ti| ti.ggml_dtype).collect();

        // 9. Build tensor manifest for fingerprint (name, dtype, shape).
        let tensor_manifest: Vec<(String, GgmlDType, Vec<usize>)> = tensor_infos
            .iter()
            .map(|(name, ti)| (name.clone(), ti.ggml_dtype, ti.shape.dims().to_vec()))
            .collect();

        let fingerprint = ModelFingerprint::compute(
            architecture,
            &metadata,
            &tensor_manifest,
            file_size,
        );

        // 10. Return aggregated errors or success.
        errs.into_result()?;

        Ok(Self {
            architecture,
            block_count,
            hidden_size,
            context_length,
            full_attention_interval,
            feed_forward_length,
            attention_head_count,
            attention_head_count_kv,
            attention_key_length,
            attention_value_length,
            rope_freq_base,
            rms_norm_eps,
            routed_experts,
            experts_per_token,
            routed_intermediate,
            shared_intermediate,
            shared_expert_count,
            router_norm_topk,
            quant_set,
            fingerprint,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_content() -> Content {
        Content {
            magic: candle_core::quantized::gguf_file::VersionedMagic::GgufV2,
            metadata: std::collections::HashMap::new(),
            tensor_infos: std::collections::HashMap::new(),
            tensor_data_offset: 0,
        }
    }

    #[test]
    fn test_missing_architecture_fails() {
        let ct = empty_content();
        let result = ModelProfile::read_and_validate(&ct, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("general.architecture"));
    }

    #[test]
    fn test_unsupported_architecture_fails() {
        let mut ct = empty_content();
        ct.metadata.insert(
            "general.architecture".into(),
            Value::String("llama".into()),
        );
        let result = ModelProfile::read_and_validate(&ct, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("unsupported architecture"));
    }

    #[test]
    fn test_dense_missing_metadata_aggregates_errors() {
        let mut ct = empty_content();
        ct.metadata.insert(
            "general.architecture".into(),
            Value::String("qwen35".into()),
        );
        // Deliberately missing all qwen35.* keys.
        let result = ModelProfile::read_and_validate(&ct, 0);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        // Should report multiple missing keys at once.
        assert!(msg.contains("qwen35.block_count"));
        assert!(msg.contains("qwen35.embedding_length"));
        assert!(msg.contains("multiple") || msg.contains("error(s)"));
    }
}