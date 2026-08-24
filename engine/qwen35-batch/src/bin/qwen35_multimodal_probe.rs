use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use qwen35_batch::real::multimodal::{
    build_position_plan, process_image, process_video, DecodedRgb, GridThw, PackedMedia,
};
use qwen35_batch::real::tokenizer::{self, ChatContent, MultimodalChatMsg};
use serde_json::{json, Value};

const TRANSFORMERS_REVISION: &str = "00e8e49eb3eda67290f635f6bdf59f236f6adf7e";

fn usage() -> ! {
    eprintln!(
        "usage:\n  qwen35_multimodal_probe image <gguf> <rgb-file> <width> <height> <prompt>\n  qwen35_multimodal_probe video <gguf> <width> <height> <source-fps> <indices-csv> <prompt> <rgb-file>..."
    );
    std::process::exit(2);
}

fn read_rgb(path: &Path, width: u32, height: u32) -> Result<DecodedRgb> {
    let bytes = std::fs::read(path).with_context(|| format!("read RGB file {}", path.display()))?;
    DecodedRgb::new(width, height, bytes)
}

fn packed_json(media: &PackedMedia) -> Result<Value> {
    let row_size = 3 * 2 * 16 * 16;
    Ok(json!({
        "grid_thw": [media.grid.t, media.grid.h, media.grid.w],
        "patch_shape": [media.grid.patch_count()?, row_size],
        "patch_values": media.patches,
        "visual_tokens": media.visual_tokens()?,
        "frame_indices": media.frame_indices,
        "timestamps": media.timestamps,
    }))
}

fn prompt_json(
    gguf: &Path,
    prompt: &str,
    content: &[ChatContent<'_>],
    image_grids: &[GridThw],
    video_grids: &[GridThw],
) -> Result<Value> {
    let tokenizer = tokenizer::load_from_gguf_path(gguf)?;
    let message = MultimodalChatMsg {
        role: "user",
        content,
    };
    let chatml = tokenizer::build_chatml_multimodal(&[message])?;
    let encoded = tokenizer::encode_multimodal_no_think(&tokenizer, &chatml)?;
    let positions = build_position_plan(&encoded.mm_token_types, image_grids, video_grids)?;
    Ok(json!({
        "input_ids": encoded.ids,
        "mm_token_type_ids": encoded.mm_token_types,
        "marker_order": chatml
            .match_indices("<|vision_start|>")
            .map(|(index, _)| index)
            .collect::<Vec<_>>(),
        "text_positions": positions.text_positions,
        "rope_positions": positions.rope_positions,
        "decode_rope_delta": positions.decode_rope_delta,
        "prompt": prompt,
    }))
}

fn parse_indices(value: &str) -> Result<Vec<usize>> {
    if value.is_empty() {
        bail!("frame index list is empty");
    }
    value
        .split(',')
        .map(|part| part.parse::<usize>().context("invalid frame index"))
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        usage();
    }
    let result = match args[1].as_str() {
        "image" if args.len() == 7 => {
            let gguf = PathBuf::from(&args[2]);
            let width = args[4].parse::<u32>().context("invalid width")?;
            let height = args[5].parse::<u32>().context("invalid height")?;
            let media = process_image(&read_rgb(Path::new(&args[3]), width, height)?)?;
            let prompt = &args[6];
            let content = [
                ChatContent::Image {
                    visual_tokens: media.visual_tokens()?,
                },
                ChatContent::Text(prompt),
            ];
            json!({
                "schema_version": "qwen35-multimodal-probe-v1",
                "transformers_revision": TRANSFORMERS_REVISION,
                "kind": "image",
                "media": packed_json(&media)?,
                "prompt": prompt_json(&gguf, prompt, &content, &[media.grid], &[])?,
            })
        }
        "video" if args.len() >= 9 => {
            let gguf = PathBuf::from(&args[2]);
            let width = args[3].parse::<u32>().context("invalid width")?;
            let height = args[4].parse::<u32>().context("invalid height")?;
            let source_fps = args[5].parse::<f64>().context("invalid source FPS")?;
            let indices = parse_indices(&args[6])?;
            let prompt = &args[7];
            let frames = args[8..]
                .iter()
                .map(|path| read_rgb(Path::new(path), width, height))
                .collect::<Result<Vec<_>>>()?;
            let media = process_video(&frames, &indices, source_fps)?;
            let content = [
                ChatContent::Video {
                    frame_tokens: media.frame_tokens()?,
                    timestamps: &media.timestamps,
                },
                ChatContent::Text(prompt),
            ];
            json!({
                "schema_version": "qwen35-multimodal-probe-v1",
                "transformers_revision": TRANSFORMERS_REVISION,
                "kind": "video",
                "media": packed_json(&media)?,
                "prompt": prompt_json(&gguf, prompt, &content, &[], &[media.grid])?,
            })
        }
        _ => usage(),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
