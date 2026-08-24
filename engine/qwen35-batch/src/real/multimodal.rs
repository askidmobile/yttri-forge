//! Official-compatible Qwen3.5 decoded-RGB processor.
//!
//! Encoded media, EXIF, ICC and alpha handling stay in server helper. This
//! module starts at validated RGB8 and reproduces pinned Transformers resize,
//! normalization, patch order, marker expansion and Qwen3.5 MRoPE positions.

use std::fmt::Write as _;

use anyhow::{anyhow, bail, ensure, Result};
use image::RgbImage;

pub const PATCH_SIZE: usize = 16;
pub const TEMPORAL_PATCH_SIZE: usize = 2;
pub const MERGE_SIZE: usize = 2;
pub const IMAGE_MIN_PIXELS: usize = 65_536;
pub const IMAGE_MAX_PIXELS: usize = 16_777_216;
pub const VIDEO_MIN_PIXELS: usize = 4_096;
pub const VIDEO_MAX_PIXELS: usize = 25_165_824;
pub const VIDEO_TARGET_FPS: f64 = 2.0;
pub const VIDEO_MIN_FRAMES: usize = 4;
pub const VIDEO_MAX_FRAMES: usize = 768;

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";

pub const VISION_START_TOKEN_ID: u32 = 248_053;
pub const VISION_END_TOKEN_ID: u32 = 248_054;
pub const IMAGE_PAD_TOKEN_ID: u32 = 248_056;
pub const VIDEO_PAD_TOKEN_ID: u32 = 248_057;

/// Frequency-source order for Qwen3.5 interleaved MRoPE sections [11, 11, 10].
/// 0=T, 1=H, 2=W. Scalar text positions make all three sources equivalent.
pub const MROPE_DIMENSION_SOURCES: [u8; 32] = [
    0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridThw {
    pub t: usize,
    pub h: usize,
    pub w: usize,
}

impl GridThw {
    pub fn patch_count(self) -> Result<usize> {
        self.t
            .checked_mul(self.h)
            .and_then(|value| value.checked_mul(self.w))
            .ok_or_else(|| anyhow!("visual patch count overflow"))
    }

    pub fn visual_tokens(self) -> Result<usize> {
        ensure!(
            self.h % MERGE_SIZE == 0 && self.w % MERGE_SIZE == 0,
            "visual grid is not divisible by merge size"
        );
        self.patch_count()?
            .checked_div(MERGE_SIZE * MERGE_SIZE)
            .ok_or_else(|| anyhow!("visual token count overflow"))
    }

    pub fn frame_tokens(self) -> Result<usize> {
        ensure!(
            self.h % MERGE_SIZE == 0 && self.w % MERGE_SIZE == 0,
            "visual grid is not divisible by merge size"
        );
        (self.h / MERGE_SIZE)
            .checked_mul(self.w / MERGE_SIZE)
            .ok_or_else(|| anyhow!("frame token count overflow"))
    }
}

#[derive(Clone, Debug)]
pub struct DecodedRgb {
    image: RgbImage,
}

impl DecodedRgb {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self> {
        ensure!(width > 0 && height > 0, "RGB dimensions must be positive");
        let expected = usize::try_from(width)?
            .checked_mul(usize::try_from(height)?)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| anyhow!("RGB byte count overflow"))?;
        ensure!(pixels.len() == expected, "RGB byte count mismatch");
        let image = RgbImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow!("invalid RGB image buffer"))?;
        Ok(Self { image })
    }

    pub fn width(&self) -> u32 {
        self.image.width()
    }

    pub fn height(&self) -> u32 {
        self.image.height()
    }

    pub fn pixels(&self) -> &[u8] {
        self.image.as_raw()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessorConfig {
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl ProcessorConfig {
    pub const IMAGE: Self = Self {
        min_pixels: IMAGE_MIN_PIXELS,
        max_pixels: IMAGE_MAX_PIXELS,
    };
    pub const VIDEO: Self = Self {
        min_pixels: VIDEO_MIN_PIXELS,
        max_pixels: VIDEO_MAX_PIXELS,
    };
}

#[derive(Clone, Debug)]
pub struct PackedMedia {
    pub kind: MediaKind,
    pub grid: GridThw,
    /// Official rows: [T*H*W, 3*2*16*16].
    pub patches: Vec<f32>,
    /// Empty for images; selected source indices for video before odd-frame pad.
    pub frame_indices: Vec<usize>,
    /// One timestamp per temporal group, in seconds.
    pub timestamps: Vec<f64>,
}

impl PackedMedia {
    pub fn visual_tokens(&self) -> Result<usize> {
        self.grid.visual_tokens()
    }

    pub fn frame_tokens(&self) -> Result<usize> {
        self.grid.frame_tokens()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PositionPlan {
    pub text_positions: Vec<u32>,
    /// [temporal, height, width].
    pub rope_positions: [Vec<u32>; 3],
    pub decode_rope_delta: i64,
}

fn checked_factor() -> usize {
    PATCH_SIZE * MERGE_SIZE
}

fn round_ratio_ties_even(value: usize, divisor: usize) -> usize {
    let quotient = value / divisor;
    let remainder = value % divisor;
    match remainder.cmp(&(divisor - remainder)) {
        std::cmp::Ordering::Less => quotient,
        std::cmp::Ordering::Greater => quotient + 1,
        std::cmp::Ordering::Equal => quotient + (quotient & 1),
    }
}

fn round_f64_ties_even(value: f64) -> Result<usize> {
    ensure!(value.is_finite() && value >= 0.0, "invalid rounding input");
    let floor = value.floor();
    let fraction = value - floor;
    let rounded = if fraction < 0.5 {
        floor
    } else if fraction > 0.5 || (floor as u64) & 1 == 1 {
        floor + 1.0
    } else {
        floor
    };
    usize::try_from(rounded as u64).map_err(Into::into)
}

fn validate_resize_inputs(
    height: usize,
    width: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<()> {
    ensure!(height > 0 && width > 0, "media dimensions must be positive");
    ensure!(
        min_pixels > 0 && min_pixels <= max_pixels,
        "invalid pixel limits"
    );
    let long = height.max(width) as f64;
    let short = height.min(width) as f64;
    ensure!(long / short <= 200.0, "absolute aspect ratio exceeds 200");
    Ok(())
}

/// Pinned Qwen2-VL image smart resize, including Python round-to-even behavior.
pub fn image_smart_resize(height: usize, width: usize) -> Result<(usize, usize)> {
    image_smart_resize_with(height, width, ProcessorConfig::IMAGE)
}

pub fn image_smart_resize_with(
    height: usize,
    width: usize,
    config: ProcessorConfig,
) -> Result<(usize, usize)> {
    validate_resize_inputs(height, width, config.min_pixels, config.max_pixels)?;
    let factor = checked_factor();
    let mut h_bar = round_ratio_ties_even(height, factor)
        .checked_mul(factor)
        .ok_or_else(|| anyhow!("resized height overflow"))?;
    let mut w_bar = round_ratio_ties_even(width, factor)
        .checked_mul(factor)
        .ok_or_else(|| anyhow!("resized width overflow"))?;
    h_bar = h_bar.max(factor);
    w_bar = w_bar.max(factor);
    let rounded_pixels = h_bar
        .checked_mul(w_bar)
        .ok_or_else(|| anyhow!("resized pixel count overflow"))?;
    if rounded_pixels > config.max_pixels {
        let beta = ((height as f64 * width as f64) / config.max_pixels as f64).sqrt();
        h_bar = (((height as f64 / beta / factor as f64).floor() as usize) * factor).max(factor);
        w_bar = (((width as f64 / beta / factor as f64).floor() as usize) * factor).max(factor);
    } else if rounded_pixels < config.min_pixels {
        let beta = (config.min_pixels as f64 / (height as f64 * width as f64)).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or_else(|| anyhow!("resized height overflow"))?;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or_else(|| anyhow!("resized width overflow"))?;
    }
    Ok((h_bar, w_bar))
}

/// Pinned Qwen3-VL video smart resize. Pixel limits include temporal extent.
pub fn video_smart_resize(
    num_frames: usize,
    height: usize,
    width: usize,
) -> Result<(usize, usize)> {
    video_smart_resize_with(num_frames, height, width, ProcessorConfig::VIDEO)
}

pub fn video_smart_resize_with(
    num_frames: usize,
    mut height: usize,
    mut width: usize,
    config: ProcessorConfig,
) -> Result<(usize, usize)> {
    ensure!(
        num_frames >= TEMPORAL_PATCH_SIZE,
        "video frame count is below temporal patch size"
    );
    validate_resize_inputs(height, width, config.min_pixels, config.max_pixels)?;
    let factor = checked_factor();
    if height < factor || width < factor {
        let scale = (factor as f64 / height as f64).max(factor as f64 / width as f64);
        height = (height as f64 * scale) as usize;
        width = (width as f64 * scale) as usize;
    }
    validate_resize_inputs(height, width, config.min_pixels, config.max_pixels)?;
    let mut h_bar = round_ratio_ties_even(height, factor)
        .checked_mul(factor)
        .ok_or_else(|| anyhow!("resized height overflow"))?;
    let mut w_bar = round_ratio_ties_even(width, factor)
        .checked_mul(factor)
        .ok_or_else(|| anyhow!("resized width overflow"))?;
    let t_bar = round_ratio_ties_even(num_frames, TEMPORAL_PATCH_SIZE)
        .checked_mul(TEMPORAL_PATCH_SIZE)
        .ok_or_else(|| anyhow!("resized frame count overflow"))?;
    let rounded_pixels = t_bar
        .checked_mul(h_bar)
        .and_then(|value| value.checked_mul(w_bar))
        .ok_or_else(|| anyhow!("video pixel count overflow"))?;
    if rounded_pixels > config.max_pixels {
        let beta =
            ((num_frames as f64 * height as f64 * width as f64) / config.max_pixels as f64).sqrt();
        h_bar = (((height as f64 / beta / factor as f64).floor() as usize) * factor).max(factor);
        w_bar = (((width as f64 / beta / factor as f64).floor() as usize) * factor).max(factor);
    } else if rounded_pixels < config.min_pixels {
        let beta =
            (config.min_pixels as f64 / (num_frames as f64 * height as f64 * width as f64)).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or_else(|| anyhow!("resized height overflow"))?;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize)
            .checked_mul(factor)
            .ok_or_else(|| anyhow!("resized width overflow"))?;
    }
    Ok((h_bar, w_bar))
}

fn keys_cubic(value: f64) -> f64 {
    let x = value.abs();
    let a = -0.5;
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
    } else {
        0.0
    }
}

#[derive(Debug)]
struct AxisWeights {
    start: usize,
    values: Vec<i16>,
}

fn quantized_axis_weights(input: usize, output: usize) -> Result<(Vec<AxisWeights>, u32)> {
    ensure!(
        input > 0 && output > 0,
        "resize dimensions must be positive"
    );
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let max_size = ((support.ceil() as usize) * 2 + 1).max(1);
    let mut raw = Vec::with_capacity(output);
    let mut max_weight = 0.0f64;
    for index in 0..output {
        let center = scale * (index as f64 + 0.5);
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(input);
        let size = end.saturating_sub(start).min(max_size);
        ensure!(size > 0, "resize produced an empty filter");
        let mut values = Vec::with_capacity(size);
        let mut total = 0.0;
        for source in start..start + size {
            let weight = keys_cubic((source as f64 - center + 0.5) * inverse_scale);
            values.push(weight);
            total += weight;
        }
        ensure!(total != 0.0, "resize filter has zero weight");
        for value in &mut values {
            *value /= total;
            max_weight = max_weight.max(*value);
        }
        raw.push((start, values));
    }
    let mut precision = 0u32;
    while precision < 22 {
        let next = (0.5 + max_weight * ((1u64 << (precision + 1)) as f64)) as i64;
        if next >= 1 << 15 {
            break;
        }
        precision += 1;
    }
    ensure!(precision > 0, "invalid resize weight precision");
    let scale_i = (1u64 << precision) as f64;
    let quantized = raw
        .into_iter()
        .map(|(start, values)| AxisWeights {
            start,
            values: values
                .into_iter()
                .map(|value| {
                    let scaled = value * scale_i;
                    if scaled < 0.0 {
                        (scaled - 0.5) as i16
                    } else {
                        (scaled + 0.5) as i16
                    }
                })
                .collect(),
        })
        .collect();
    Ok((quantized, precision))
}

/// Torchvision/PyTorch uint8 bicubic antialias resize used by pinned fast processor.
/// Horizontal and vertical passes both use quantized weights and uint8 rounding.
pub fn resize_rgb_exact(input: &DecodedRgb, width: usize, height: usize) -> Result<DecodedRgb> {
    ensure!(
        width > 0 && height > 0,
        "resize dimensions must be positive"
    );
    let source_width = usize::try_from(input.width())?;
    let source_height = usize::try_from(input.height())?;
    if source_width == width && source_height == height {
        return DecodedRgb::new(input.width(), input.height(), input.pixels().to_vec());
    }
    let (horizontal, horizontal_precision) = quantized_axis_weights(source_width, width)?;
    let horizontal_len = source_height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| anyhow!("resize buffer overflow"))?;
    let mut temp = vec![0u8; horizontal_len];
    let horizontal_round = 1i64 << (horizontal_precision - 1);
    for y in 0..source_height {
        for (x, weights) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let mut value = horizontal_round;
                for (offset, weight) in weights.values.iter().enumerate() {
                    let source =
                        input.pixels()[((y * source_width + weights.start + offset) * 3) + channel];
                    value += i64::from(source) * i64::from(*weight);
                }
                temp[(y * width + x) * 3 + channel] =
                    (value >> horizontal_precision).clamp(0, 255) as u8;
            }
        }
    }

    let (vertical, vertical_precision) = quantized_axis_weights(source_height, height)?;
    let output_len = height
        .checked_mul(width)
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| anyhow!("resize buffer overflow"))?;
    let mut output = vec![0u8; output_len];
    let vertical_round = 1i64 << (vertical_precision - 1);
    for (y, weights) in vertical.iter().enumerate() {
        for x in 0..width {
            for channel in 0..3 {
                let mut value = vertical_round;
                for (offset, weight) in weights.values.iter().enumerate() {
                    let source = temp[(((weights.start + offset) * width + x) * 3) + channel];
                    value += i64::from(source) * i64::from(*weight);
                }
                output[(y * width + x) * 3 + channel] =
                    (value >> vertical_precision).clamp(0, 255) as u8;
            }
        }
    }
    DecodedRgb::new(u32::try_from(width)?, u32::try_from(height)?, output)
}

#[inline]
fn normalize(value: u8) -> f32 {
    (f32::from(value) - 127.5) / 127.5
}

fn packed_capacity(grid: GridThw) -> Result<usize> {
    grid.patch_count()?
        .checked_mul(3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)
        .ok_or_else(|| anyhow!("packed patch buffer overflow"))
}

pub fn process_image(image: &DecodedRgb) -> Result<PackedMedia> {
    process_image_with(image, ProcessorConfig::IMAGE)
}

pub fn process_image_with(image: &DecodedRgb, config: ProcessorConfig) -> Result<PackedMedia> {
    let (height, width) = image_smart_resize_with(
        usize::try_from(image.height())?,
        usize::try_from(image.width())?,
        config,
    )?;
    let resized = resize_rgb_exact(image, width, height)?;
    let grid = GridThw {
        t: 1,
        h: height / PATCH_SIZE,
        w: width / PATCH_SIZE,
    };
    let temporal_pair = [resized.clone(), resized];
    let mut patches = Vec::with_capacity(packed_capacity(grid)?);
    pack_frames(&temporal_pair, grid, &mut patches)?;
    Ok(PackedMedia {
        kind: MediaKind::Image,
        grid,
        patches,
        frame_indices: Vec::new(),
        timestamps: Vec::new(),
    })
}

pub fn process_video(
    frames: &[DecodedRgb],
    frame_indices: &[usize],
    source_fps: f64,
) -> Result<PackedMedia> {
    process_video_with(frames, frame_indices, source_fps, ProcessorConfig::VIDEO)
}

pub fn process_video_with(
    frames: &[DecodedRgb],
    frame_indices: &[usize],
    source_fps: f64,
    config: ProcessorConfig,
) -> Result<PackedMedia> {
    ensure!(
        frames.len() >= TEMPORAL_PATCH_SIZE,
        "video has too few frames"
    );
    ensure!(
        frames.len() == frame_indices.len(),
        "video frame index count mismatch"
    );
    ensure!(
        source_fps.is_finite() && source_fps > 0.0,
        "invalid source FPS"
    );
    let source_width = frames[0].width();
    let source_height = frames[0].height();
    ensure!(
        frames
            .iter()
            .all(|frame| frame.width() == source_width && frame.height() == source_height),
        "video frames have different dimensions"
    );
    let (height, width) = video_smart_resize_with(
        frames.len(),
        usize::try_from(source_height)?,
        usize::try_from(source_width)?,
        config,
    )?;
    let mut resized = Vec::with_capacity(frames.len() + 1);
    for frame in frames {
        resized.push(resize_rgb_exact(frame, width, height)?);
    }
    if resized.len() % TEMPORAL_PATCH_SIZE != 0 {
        resized.push(
            resized
                .last()
                .cloned()
                .ok_or_else(|| anyhow!("video has no frames"))?,
        );
    }
    let grid = GridThw {
        t: resized.len() / TEMPORAL_PATCH_SIZE,
        h: height / PATCH_SIZE,
        w: width / PATCH_SIZE,
    };
    let mut patches = Vec::with_capacity(packed_capacity(grid)?);
    pack_frames(&resized, grid, &mut patches)?;
    Ok(PackedMedia {
        kind: MediaKind::Video,
        grid,
        patches,
        frame_indices: frame_indices.to_vec(),
        timestamps: calculate_timestamps(frame_indices, source_fps)?,
    })
}

fn pack_frames(frames: &[DecodedRgb], grid: GridThw, output: &mut Vec<f32>) -> Result<()> {
    let expected_frames = grid
        .t
        .checked_mul(TEMPORAL_PATCH_SIZE)
        .ok_or_else(|| anyhow!("temporal grid overflow"))?;
    ensure!(
        frames.len() == expected_frames,
        "temporal frame count mismatch"
    );
    let height = grid.h * PATCH_SIZE;
    let width = grid.w * PATCH_SIZE;
    ensure!(
        frames
            .iter()
            .all(|frame| { frame.width() as usize == width && frame.height() as usize == height }),
        "frame dimensions do not match visual grid"
    );
    ensure!(
        grid.h % MERGE_SIZE == 0 && grid.w % MERGE_SIZE == 0,
        "visual grid is not divisible by merge size"
    );

    for temporal in 0..grid.t {
        for block_h in 0..grid.h / MERGE_SIZE {
            for block_w in 0..grid.w / MERGE_SIZE {
                for merge_h in 0..MERGE_SIZE {
                    for merge_w in 0..MERGE_SIZE {
                        let patch_h = block_h * MERGE_SIZE + merge_h;
                        let patch_w = block_w * MERGE_SIZE + merge_w;
                        for channel in 0..3 {
                            for temporal_inner in 0..TEMPORAL_PATCH_SIZE {
                                let frame =
                                    &frames[temporal * TEMPORAL_PATCH_SIZE + temporal_inner];
                                for y in 0..PATCH_SIZE {
                                    for x in 0..PATCH_SIZE {
                                        let pixel_y = patch_h * PATCH_SIZE + y;
                                        let pixel_x = patch_w * PATCH_SIZE + x;
                                        output.push(normalize(
                                            frame.pixels()
                                                [(pixel_y * width + pixel_x) * 3 + channel],
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ensure!(
        output.len() == packed_capacity(grid)?,
        "packed patch size mismatch"
    );
    Ok(())
}

/// Pinned `np.linspace(...).round().astype(int)` frame selection.
pub fn sample_frame_indices(total_frames: usize, source_fps: f64) -> Result<Vec<usize>> {
    ensure!(total_frames > 0, "video has no frames");
    ensure!(
        source_fps.is_finite() && source_fps > 0.0,
        "invalid source FPS"
    );
    let requested = ((total_frames as f64 / source_fps) * VIDEO_TARGET_FPS).floor() as usize;
    let count = requested
        .max(VIDEO_MIN_FRAMES)
        .min(VIDEO_MAX_FRAMES)
        .min(total_frames);
    linspace_indices(total_frames, count)
}

pub fn linspace_indices(total_frames: usize, count: usize) -> Result<Vec<usize>> {
    ensure!(
        total_frames > 0 && count > 0 && count <= total_frames,
        "invalid frame sample size"
    );
    if count == 1 {
        return Ok(vec![0]);
    }
    let end = (total_frames - 1) as f64;
    (0..count)
        .map(|index| round_f64_ties_even(end * index as f64 / (count - 1) as f64))
        .collect()
}

/// Highest deterministic even frame count fitting visual-token admission.
pub fn select_video_frame_indices(
    total_frames: usize,
    source_fps: f64,
    height: usize,
    width: usize,
    max_visual_tokens: usize,
) -> Result<Vec<usize>> {
    let requested = sample_frame_indices(total_frames, source_fps)?.len();
    let mut count = requested - (requested % TEMPORAL_PATCH_SIZE);
    while count >= TEMPORAL_PATCH_SIZE {
        let (resized_h, resized_w) = video_smart_resize(count, height, width)?;
        let grid = GridThw {
            t: count / TEMPORAL_PATCH_SIZE,
            h: resized_h / PATCH_SIZE,
            w: resized_w / PATCH_SIZE,
        };
        if grid.visual_tokens()? <= max_visual_tokens {
            return linspace_indices(total_frames, count);
        }
        count -= TEMPORAL_PATCH_SIZE;
    }
    bail!("video exceeds visual-token budget")
}

pub fn calculate_timestamps(indices: &[usize], source_fps: f64) -> Result<Vec<f64>> {
    ensure!(!indices.is_empty(), "video has no frame indices");
    ensure!(
        source_fps.is_finite() && source_fps > 0.0,
        "invalid source FPS"
    );
    let mut padded = indices.to_vec();
    while padded.len() % TEMPORAL_PATCH_SIZE != 0 {
        padded.push(*padded.last().unwrap());
    }
    Ok(padded
        .chunks_exact(TEMPORAL_PATCH_SIZE)
        .map(|pair| (pair[0] as f64 + pair[TEMPORAL_PATCH_SIZE - 1] as f64) / (2.0 * source_fps))
        .collect())
}

pub fn image_marker(visual_tokens: usize) -> Result<String> {
    ensure!(visual_tokens > 0, "image has no visual tokens");
    let mut marker = String::with_capacity(
        VISION_START.len() + VISION_END.len() + visual_tokens.saturating_mul(IMAGE_PAD.len()),
    );
    marker.push_str(VISION_START);
    for _ in 0..visual_tokens {
        marker.push_str(IMAGE_PAD);
    }
    marker.push_str(VISION_END);
    Ok(marker)
}

pub fn video_marker(frame_tokens: usize, timestamps: &[f64]) -> Result<String> {
    ensure!(frame_tokens > 0, "video frame has no visual tokens");
    ensure!(!timestamps.is_empty(), "video has no timestamps");
    ensure!(
        timestamps.iter().all(|value| value.is_finite()),
        "video timestamp is not finite"
    );
    let mut marker = String::new();
    for timestamp in timestamps {
        write!(&mut marker, "<{timestamp:.1} seconds>")?;
        marker.push_str(VISION_START);
        for _ in 0..frame_tokens {
            marker.push_str(VIDEO_PAD);
        }
        marker.push_str(VISION_END);
    }
    Ok(marker)
}

fn push_text_positions(start: u32, len: usize, rope: &mut [Vec<u32>; 3]) -> Result<u32> {
    for offset in 0..len {
        let value = start
            .checked_add(u32::try_from(offset)?)
            .ok_or_else(|| anyhow!("position overflow"))?;
        for dimension in rope.iter_mut() {
            dimension.push(value);
        }
    }
    start
        .checked_add(u32::try_from(len)?)
        .ok_or_else(|| anyhow!("position overflow"))
}

fn push_vision_positions(start: u32, grid: GridThw, rope: &mut [Vec<u32>; 3]) -> Result<u32> {
    ensure!(
        grid.h % MERGE_SIZE == 0 && grid.w % MERGE_SIZE == 0,
        "visual grid is not divisible by merge size"
    );
    let llm_h = grid.h / MERGE_SIZE;
    let llm_w = grid.w / MERGE_SIZE;
    for temporal in 0..grid.t {
        for height in 0..llm_h {
            for width in 0..llm_w {
                rope[0].push(
                    start
                        .checked_add(u32::try_from(temporal)?)
                        .ok_or_else(|| anyhow!("position overflow"))?,
                );
                rope[1].push(
                    start
                        .checked_add(u32::try_from(height)?)
                        .ok_or_else(|| anyhow!("position overflow"))?,
                );
                rope[2].push(
                    start
                        .checked_add(u32::try_from(width)?)
                        .ok_or_else(|| anyhow!("position overflow"))?,
                );
            }
        }
    }
    start
        .checked_add(u32::try_from(llm_h.max(llm_w))?)
        .ok_or_else(|| anyhow!("position overflow"))
}

/// Construct official Qwen3.5 positions from tokenizer `mm_token_type_ids`.
/// Video grids are split into one grid per timestamp-separated temporal group.
pub fn build_position_plan(
    mm_token_types: &[u8],
    image_grids: &[GridThw],
    video_grids: &[GridThw],
) -> Result<PositionPlan> {
    ensure!(
        mm_token_types.iter().all(|kind| *kind <= 2),
        "unknown multimodal token type"
    );
    let mut expanded_video_grids = Vec::new();
    for grid in video_grids {
        ensure!(grid.t > 0, "video grid has no temporal groups");
        for _ in 0..grid.t {
            expanded_video_grids.push(GridThw {
                t: 1,
                h: grid.h,
                w: grid.w,
            });
        }
    }
    let mut image_index = 0usize;
    let mut video_index = 0usize;
    let mut current = 0u32;
    let mut rope = [Vec::new(), Vec::new(), Vec::new()];
    let mut index = 0usize;
    while index < mm_token_types.len() {
        let kind = mm_token_types[index];
        let end = mm_token_types[index..]
            .iter()
            .position(|candidate| *candidate != kind)
            .map(|offset| index + offset)
            .unwrap_or(mm_token_types.len());
        let len = end - index;
        match kind {
            0 => current = push_text_positions(current, len, &mut rope)?,
            1 => {
                let grid = *image_grids
                    .get(image_index)
                    .ok_or_else(|| anyhow!("missing image grid"))?;
                ensure!(grid.visual_tokens()? == len, "image token/grid mismatch");
                current = push_vision_positions(current, grid, &mut rope)?;
                image_index += 1;
            }
            2 => {
                let grid = *expanded_video_grids
                    .get(video_index)
                    .ok_or_else(|| anyhow!("missing video grid"))?;
                ensure!(grid.visual_tokens()? == len, "video token/grid mismatch");
                current = push_vision_positions(current, grid, &mut rope)?;
                video_index += 1;
            }
            _ => unreachable!(),
        }
        index = end;
    }
    ensure!(image_index == image_grids.len(), "unused image grid");
    ensure!(
        video_index == expanded_video_grids.len(),
        "unused video grid"
    );
    let max_position = rope
        .iter()
        .flat_map(|dimension| dimension.iter())
        .copied()
        .max()
        .map_or(0i64, |value| i64::from(value) + 1);
    Ok(PositionPlan {
        text_positions: (0..mm_token_types.len())
            .map(u32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?,
        rope_positions: rope,
        decode_rope_delta: max_position - i64::try_from(mm_token_types.len())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mrope_sections_are_11_11_10() {
        assert_eq!(
            MROPE_DIMENSION_SOURCES.iter().filter(|v| **v == 0).count(),
            11
        );
        assert_eq!(
            MROPE_DIMENSION_SOURCES.iter().filter(|v| **v == 1).count(),
            11
        );
        assert_eq!(
            MROPE_DIMENSION_SOURCES.iter().filter(|v| **v == 2).count(),
            10
        );
    }

    #[test]
    fn python_rounding_is_ties_even() {
        assert_eq!(round_ratio_ties_even(80, 32), 2);
        assert_eq!(round_ratio_ties_even(112, 32), 4);
        assert_eq!(linspace_indices(4, 3).unwrap(), vec![0, 2, 3]);
    }

    #[test]
    fn official_image_resize_keeps_fixture_shape() {
        assert_eq!(image_smart_resize(384, 640).unwrap(), (384, 640));
    }
}
