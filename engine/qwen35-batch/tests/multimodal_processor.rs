use qwen35_batch::real::multimodal::{
    build_position_plan, calculate_timestamps, image_marker, image_smart_resize_with,
    linspace_indices, process_image_with, process_video_with, resize_rgb_exact, video_marker,
    DecodedRgb, GridThw, ProcessorConfig, IMAGE_PAD, VIDEO_PAD,
};
use qwen35_batch::real::tokenizer::{build_chatml_multimodal, ChatContent, MultimodalChatMsg};

fn synthetic_rgb(width: u32, height: u32, seed: u8) -> DecodedRgb {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x as u8).wrapping_add(seed));
            pixels.push((y as u8).wrapping_mul(3).wrapping_add(seed));
            pixels.push((x as u8).wrapping_add(y as u8).wrapping_add(seed));
        }
    }
    DecodedRgb::new(width, height, pixels).unwrap()
}

#[test]
fn resize_matches_pinned_torchvision_uint8_vector() {
    let chw = (0..60u8).collect::<Vec<_>>();
    let mut pixels = Vec::with_capacity(60);
    for pixel in 0..20 {
        pixels.push(chw[pixel]);
        pixels.push(chw[20 + pixel]);
        pixels.push(chw[40 + pixel]);
    }
    let image = DecodedRgb::new(5, 4, pixels).unwrap();
    let resized = resize_rgb_exact(&image, 8, 8).unwrap();
    let expected = [
        0, 0, 1, 2, 2, 3, 4, 4, 1, 1, 2, 3, 3, 4, 5, 5, 4, 4, 5, 6, 6, 7, 8, 8, 6, 6, 7, 8, 8, 9,
        10, 10, 9, 9, 10, 11, 11, 12, 13, 13, 11, 11, 12, 13, 13, 14, 15, 15, 14, 14, 15, 16, 16,
        17, 18, 18, 15, 15, 16, 17, 17, 18, 19, 19, 20, 20, 21, 22, 22, 23, 24, 24, 21, 21, 22, 23,
        23, 24, 25, 25, 24, 24, 25, 26, 26, 27, 28, 28, 26, 26, 27, 28, 28, 29, 30, 30, 29, 29, 30,
        31, 31, 32, 33, 33, 31, 31, 32, 33, 33, 34, 35, 35, 34, 34, 35, 36, 36, 37, 38, 38, 35, 35,
        36, 37, 37, 38, 39, 39, 40, 40, 41, 42, 42, 43, 44, 44, 41, 41, 42, 43, 43, 44, 45, 45, 44,
        44, 45, 46, 46, 47, 48, 48, 46, 46, 47, 48, 48, 49, 50, 50, 49, 49, 50, 51, 51, 52, 53, 53,
        51, 51, 52, 53, 53, 54, 55, 55, 54, 54, 55, 56, 56, 57, 58, 58, 55, 55, 56, 57, 57, 58, 59,
        59,
    ];
    let mut expected_hwc = Vec::with_capacity(expected.len());
    for pixel in 0..64 {
        expected_hwc.push(expected[pixel]);
        expected_hwc.push(expected[64 + pixel]);
        expected_hwc.push(expected[128 + pixel]);
    }
    assert_eq!(resized.pixels(), expected_hwc);
}

#[test]
fn image_patch_order_and_values_are_official() {
    let config = ProcessorConfig {
        min_pixels: 32 * 32,
        max_pixels: 32 * 32,
    };
    let image = process_image_with(&synthetic_rgb(32, 32, 0), config).unwrap();
    assert_eq!(image.grid, GridThw { t: 1, h: 2, w: 2 });
    assert_eq!(image.visual_tokens().unwrap(), 1);
    assert_eq!(image.patches.len(), 4 * 3 * 2 * 16 * 16);

    let norm = |value: u8| (f32::from(value) - 127.5) / 127.5;
    // First row: merge(0,0), channel R, temporal duplicate, patch row-major.
    assert_eq!(image.patches[0], norm(0));
    assert_eq!(image.patches[1], norm(1));
    assert_eq!(image.patches[15], norm(15));
    assert_eq!(image.patches[16], norm(0));
    assert_eq!(image.patches[256], norm(0));
    // Next channel after two temporal copies.
    assert_eq!(image.patches[512], norm(0));
}

#[test]
fn sampling_timestamps_and_video_markers_match_reference() {
    assert_eq!(linspace_indices(10, 4).unwrap(), vec![0, 3, 6, 9]);
    let timestamps = calculate_timestamps(&[0, 3, 6], 3.0).unwrap();
    assert_eq!(timestamps, vec![0.5, 2.0]);
    let marker = video_marker(2, &timestamps).unwrap();
    assert_eq!(
        marker,
        format!(
            "<0.5 seconds><|vision_start|>{0}{0}<|vision_end|><2.0 seconds><|vision_start|>{0}{0}<|vision_end|>",
            VIDEO_PAD
        )
    );

    let config = ProcessorConfig {
        min_pixels: 4 * 32 * 32,
        max_pixels: 4 * 32 * 32,
    };
    let frames = vec![
        synthetic_rgb(32, 32, 0),
        synthetic_rgb(32, 32, 1),
        synthetic_rgb(32, 32, 2),
    ];
    let video = process_video_with(&frames, &[0, 3, 6], 3.0, config).unwrap();
    assert_eq!(video.grid, GridThw { t: 2, h: 2, w: 2 });
    assert_eq!(video.visual_tokens().unwrap(), 2);
    assert_eq!(video.timestamps, timestamps);
}

#[test]
fn mixed_chatml_preserves_english_and_russian_order() {
    let image_tokens = 2;
    let timestamps = [0.5, 1.5];
    let content = [
        ChatContent::Text("English "),
        ChatContent::Image {
            visual_tokens: image_tokens,
        },
        ChatContent::Text(" русский "),
        ChatContent::Video {
            frame_tokens: 1,
            timestamps: &timestamps,
        },
        ChatContent::Text(" конец"),
    ];
    let prompt = build_chatml_multimodal(&[MultimodalChatMsg {
        role: "user",
        content: &content,
    }])
    .unwrap();
    let image = image_marker(image_tokens).unwrap();
    let video = video_marker(1, &timestamps).unwrap();
    assert_eq!(
        prompt,
        format!(
            "<|im_start|>user\nEnglish {image} русский {video} конец<|im_end|>\n<|im_start|>assistant\n"
        )
    );
    assert_eq!(prompt.matches(IMAGE_PAD).count(), 2);
    assert_eq!(prompt.matches(VIDEO_PAD).count(), 2);
}

#[test]
fn mrope_positions_and_decode_delta_match_qwen35_reference() {
    // text(2), image grid 1x4x4 -> 4 LLM tokens, text(1),
    // video grid 2x2x4 -> two timestamp-separated groups of 2 tokens, text(1).
    let types = [0, 0, 1, 1, 1, 1, 0, 2, 2, 0, 2, 2, 0];
    let plan = build_position_plan(
        &types,
        &[GridThw { t: 1, h: 4, w: 4 }],
        &[GridThw { t: 2, h: 2, w: 4 }],
    )
    .unwrap();
    assert_eq!(plan.text_positions, (0..13).collect::<Vec<_>>());
    assert_eq!(
        plan.rope_positions,
        [
            vec![0, 1, 2, 2, 2, 2, 4, 5, 5, 7, 8, 8, 10],
            vec![0, 1, 2, 2, 3, 3, 4, 5, 5, 7, 8, 8, 10],
            vec![0, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ]
    );
    assert_eq!(plan.decode_rope_delta, -2);
}

#[test]
fn malformed_inputs_fail_closed() {
    assert!(DecodedRgb::new(2, 2, vec![0; 11]).is_err());
    assert!(image_smart_resize_with(
        1,
        1000,
        ProcessorConfig {
            min_pixels: 1024,
            max_pixels: 1024,
        }
    )
    .is_err());
    assert!(build_position_plan(&[1], &[], &[]).is_err());
    assert!(build_chatml_multimodal(&[MultimodalChatMsg {
        role: "system",
        content: &[ChatContent::Image { visual_tokens: 1 }],
    }])
    .is_err());
}
