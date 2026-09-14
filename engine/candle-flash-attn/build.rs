// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use cudaforge::{KernelBuilder, Result};
use std::{fs, path::PathBuf};
const CUTLASS_COMMIT: &str = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";

const KERNEL_FILES: [&str; 53] = [
    "kernels/flash_api.cu",
    "kernels/flash_fwd_splitkv_hdim512_fp16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim512_bf16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim512_fp16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim512_bf16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_splitkv_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_hdim512_fp16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_sm80.cu",
    "kernels/flash_fwd_hdim512_bf16_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim512_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim128_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim512_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_bf16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_bf16_causal_sm80.cu",
];

const HEADER_FILES: [&str; 18] = [
    "kernels/alibi.h",
    "kernels/block_info.h",
    "kernels/dropout.h",
    "kernels/error.h",
    "kernels/flash.h",
    "kernels/flash_fwd_kernel.h",
    "kernels/flash_fwd_launch_template.h",
    "kernels/hardware_info.h",
    "kernels/kernel_helpers.h",
    "kernels/kernel_traits.h",
    "kernels/kernel_traits_sm90.h",
    "kernels/kernels.h",
    "kernels/mask.h",
    "kernels/philox.cuh",
    "kernels/rotary.h",
    "kernels/softmax.h",
    "kernels/static_switch.h",
    "kernels/utils.h",
];

/// Каталог библиотек CUDA Toolkit для СТАТИЧЕСКОЙ линковки cudart.
///
/// Динамический `-lcudart` резолвил линкер через `LIBRARY_PATH` (Linux) или
/// `LIB` (MSVC), поэтому build.rs про CUDA ничего знать не требовалось. Для
/// `rustc-link-lib=static=` файл ищет САМ rustc по своим `link-search`-путям
/// и падает раньше линкера — значит путь нужно сообщить явно.
///
/// Корень берём из окружения: его выставляют и `build_linux.sh`, и
/// `with_msvc.ps1` (там же пинится нужная версия тулкита).
fn cuda_lib_dir(is_target_msvc: bool) -> Option<PathBuf> {
    let root = ["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME", "CUDA_TOOLKIT_ROOT_DIR"]
        .iter()
        .find_map(std::env::var_os)
        .map(PathBuf::from)?;

    let candidates: Vec<PathBuf> = if is_target_msvc {
        vec![root.join("lib").join("x64")]
    } else {
        vec![
            root.join("lib64"),
            root.join("targets").join("x86_64-linux").join("lib"),
            root.join("lib"),
        ]
    };
    let stem = if is_target_msvc {
        "cudart_static.lib"
    } else {
        "libcudart_static.a"
    };
    candidates.into_iter().find(|dir| dir.join(stem).is_file())
}

fn update_hash(hash: &mut u64, bytes: &[u8]) {
    const FNV_PRIME: u64 = 1099511628211;
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn header_hash() -> Result<u64> {
    let mut hash = 14695981039346656037;
    for file in HEADER_FILES {
        update_hash(&mut hash, file.as_bytes());
        update_hash(&mut hash, &fs::read(file)?);
    }
    Ok(hash)
}

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo::rerun-if-changed={kernel_file}");
    }
    for header_file in HEADER_FILES.iter() {
        println!("cargo::rerun-if-changed={header_file}");
    }
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };

    let kernels: Vec<_> = KERNEL_FILES.iter().collect();
    let header_hash_arg = format!("-DCANDLE_FLASH_ATTN_HEADER_HASH=0x{:016x}", header_hash()?);
    let mut builder = KernelBuilder::new()
        .source_files(kernels)
        .out_dir(&build_dir)
        .with_cutlass(Some(CUTLASS_COMMIT)) // ✅ Auto-fetch and include CUTLASS from GitHub
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        .arg(&header_hash_arg)
        .thread_percentage(0.5); // Use up to 50% of available threads

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let out_file = build_dir.join("libflashattention.a");
    builder.build_lib(out_file)?;

    println!("cargo::rustc-link-search={}", build_dir.display());
    println!("cargo::rustc-link-lib=flashattention");

    // cudart линкуется СТАТИЧЕСКИ, а не как dylib.
    //
    // Динамический вариант кладёт в итоговый бинарь жёсткую зависимость —
    // импорт `cudart64_12.dll` на Windows, `DT_NEEDED libcudart.so.12` на
    // Linux, — и загрузчик резолвит её ДО main(). На машине без CUDA-рантайма
    // приложение не стартует вовсе: ни окна с ошибкой, ни строки в логах,
    // даже если GPU не нужен и код FlashAttention никогда не вызывается.
    // Так умерла бета Yttri 0.89.3-beta.1 на Windows и Linux (2026-09-14).
    //
    // Статический cudart сам грузит драйвер (`nvcuda.dll` / `libcuda.so.1`)
    // лениво, поэтому бинарь остаётся запускаемым везде, а CUDA-функции
    // отказывают в рантайме — это и есть ожидаемое поведение.
    //
    // Второго экземпляра рантайма в процессе не появляется: потребители
    // линкуют `cudarc` с `features = ["driver"]`, то есть используют
    // driver API, а не runtime API. Семантика выполнения не меняется —
    // меняется только момент и способ загрузки.
    for var in ["CUDA_PATH", "CUDA_ROOT", "CUDA_HOME", "CUDA_TOOLKIT_ROOT_DIR"] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    let Some(cuda_lib) = cuda_lib_dir(is_target_msvc) else {
        panic!(
            "cudart_static не найден: задай CUDA_PATH (или CUDA_ROOT/CUDA_HOME) \
             на корень CUDA Toolkit. Динамический cudart намеренно НЕ используется — \
             он делает бинарь незапускаемым на машине без CUDA-рантайма."
        );
    };
    println!("cargo::rustc-link-search=native={}", cuda_lib.display());
    println!("cargo::rustc-link-lib=static=cudart_static");
    if !is_target_msvc {
        // Зависимости статического cudart: динамическая загрузка драйвера и
        // таймеры. На glibc 2.34+ они слиты в libc, но стаб-библиотеки на
        // месте и линковка от их упоминания не страдает.
        println!("cargo::rustc-link-lib=dylib=dl");
        println!("cargo::rustc-link-lib=dylib=rt");
        println!("cargo::rustc-link-lib=dylib=pthread");
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
