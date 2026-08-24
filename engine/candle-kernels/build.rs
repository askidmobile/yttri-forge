use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");
    println!("cargo::rerun-if-changed=src/moe_dequant.cuh");
    // Явно следим за ВСЕМИ .cu (новый файл без этого не пересобирает ptx.rs —
    // поймано: FLASH_DECODE отсутствовал в сгенерированном модуле).
    for entry in std::fs::read_dir("src").unwrap() {
        let p = entry.unwrap().path();
        if p.extension().and_then(|s| s.to_str()) == Some("cu") {
            println!("cargo::rerun-if-changed={}", p.display());
        }
    }
    // mmq_gguf/ (MMQ Tensor-Core для плотного prefill) — следим за .cu и .cuh.
    if let Ok(rd) = std::fs::read_dir("src/mmq_gguf") {
        for entry in rd {
            let p = entry.unwrap().path();
            if matches!(p.extension().and_then(|s| s.to_str()), Some("cu") | Some("cuh")) {
                println!("cargo::rerun-if-changed={}", p.display());
            }
        }
    }

    // Build for PTX.
    // Exclude the old reference MoE kernels under src/moe/ (they use the CUDA Runtime
    // API and are not PTX-compatible). The new device-only kernels moe_router.cu and
    // moe_quantized.cu live directly under src/ and must be compiled.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let mut builder = KernelBuilder::new()
        .source_dir("src")
        .exclude(&["moe/*", "mmvq_gguf.cu", "mmq_*.cu"])
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    #[cfg(target_os = "windows")]
    {
        builder = builder.arg("-Xcompiler").arg("/Zc:preprocessor");
    }

    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            builder = builder
                .arg("-DCCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING")
                .arg("-Xcompiler")
                .arg("/Zc:preprocessor");
        }
    }

    let bindings = builder.build_ptx()?;

    // Normalize PTX version if compiled with bleeding-edge CUDA Toolkit (e.g. 13.2 writing .version 9.2)
    // so that installed NVIDIA driver JIT compilers accepting PTX ISA 8.x can load the modules seamlessly.
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "ptx") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if content.contains(".version 9.") {
                        let normalized = content
                            .replace(".version 9.2", ".version 8.6")
                            .replace(".version 9.1", ".version 8.6")
                            .replace(".version 9.0", ".version 8.6");
                        let _ = std::fs::write(&path, normalized);
                    }
                }
            }
        }
    }

    bindings.write(&ptx_path)?;
    Ok(())
}
