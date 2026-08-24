use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_file(name: &str, records: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("qwen35moe-{name}-{nonce}.jsonl"));
    fs::write(&path, records).unwrap();
    path
}

fn record(step: usize, argmax: usize, margin: f64, values: Option<&str>) -> String {
    match values {
        Some(values) => format!(
            "{{\"type\":\"logits\",\"step\":{step},\"argmax\":{argmax},\"margin\":{margin},\"finite\":3,\"values\":{values}}}\n"
        ),
        None => format!(
            "{{\"type\":\"logits\",\"step\":{step},\"argmax\":{argmax},\"margin\":{margin},\"finite\":3}}\n"
        ),
    }
}

fn tokens(count: usize) -> String {
    let ids = vec!["0"; count].join(",");
    format!("{{\"type\":\"tokens\",\"ids\":[{ids}],\"fed_ids\":[{ids}]}}\n")
}

fn run_with_args(
    reference: &std::path::Path,
    candidate: &std::path::Path,
    gate: bool,
) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qwen35moe_compare"));
    if gate {
        command.arg("--gate");
    }
    command.args([reference, candidate]).output().unwrap()
}

fn run(reference: &std::path::Path, candidate: &std::path::Path) -> std::process::Output {
    run_with_args(reference, candidate, true)
}

#[test]
fn gate_accepts_low_margin_argmax_drift_with_close_logits() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let argmax = usize::from(step == 16);
        let margin = if step == 16 { 0.29 } else { 1.0 };
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, argmax, margin, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    reference.push_str(&tokens(128));
    candidate.push_str(&tokens(128));
    let reference = temp_file("reference-pass", &reference);
    let candidate = temp_file("candidate-pass", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn gate_rejects_high_margin_argmax_drift() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, usize::from(step == 16), 0.31, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    reference.push_str(&tokens(128));
    candidate.push_str(&tokens(128));
    let reference = temp_file("reference-fail", &reference);
    let candidate = temp_file("candidate-fail", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("argmax divergence at step 16"));
}

#[test]
fn gate_rejects_large_numerical_drift() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let full = matches!(step, 16 | 45 | 50 | 92 | 111);
        reference.push_str(&record(step, 0, 1.0, full.then_some("[1.0,2.0,3.0]")));
        candidate.push_str(&record(step, 0, 1.0, full.then_some("[10.0,0.0,0.0]")));
    }
    reference.push_str(&tokens(128));
    candidate.push_str(&tokens(128));
    let reference = temp_file("reference-drift", &reference);
    let candidate = temp_file("candidate-drift", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("numerical gate failed"));
}

#[test]
fn gate_rejects_missing_required_step() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..127 {
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, 0, 1.0, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    reference.push_str(&tokens(127));
    candidate.push_str(&tokens(127));
    let reference = temp_file("reference-short", &reference);
    let candidate = temp_file("candidate-short", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("gate needs exactly 128 logits records")
    );
}

#[test]
fn comparator_rejects_empty_inputs() {
    let reference = temp_file("reference-empty", "");
    let candidate = temp_file("candidate-empty", "");
    let output = run_with_args(&reference, &candidate, false);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no logits records"));
}

#[test]
fn comparator_rejects_different_step_sets() {
    let reference = temp_file("reference-steps", &(record(0, 0, 1.0, None) + &tokens(1)));
    let candidate = temp_file("candidate-steps", &(record(1, 0, 1.0, None) + &tokens(1)));
    let output = run_with_args(&reference, &candidate, false);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("step sets differ"));
}

#[test]
fn comparator_rejects_duplicate_logits_step() {
    let duplicate = record(0, 0, 1.0, None) + &record(0, 0, 1.0, None) + &tokens(2);
    let reference = temp_file("reference-duplicate", &duplicate);
    let candidate = temp_file("candidate-duplicate", &duplicate);
    let output = run_with_args(&reference, &candidate, false);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("duplicate logits step"));
}

#[test]
fn comparator_rejects_missing_tokens() {
    let reference = temp_file("reference-no-tokens", &record(0, 0, 1.0, None));
    let candidate = temp_file("candidate-no-tokens", &record(0, 0, 1.0, None));
    let output = run_with_args(&reference, &candidate, false);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("no tokens record"));
}

#[test]
fn gate_rejects_different_teacher_forced_streams() {
    let mut reference = String::new();
    let mut candidate = String::new();
    for step in 0..128 {
        let values = matches!(step, 16 | 45 | 50 | 92 | 111).then_some("[1.0,2.0,3.0]");
        reference.push_str(&record(step, 0, 1.0, values));
        candidate.push_str(&record(step, 0, 1.0, values));
    }
    reference.push_str(&tokens(128));
    let zeros = vec!["0"; 128].join(",");
    let ones = vec!["1"; 128].join(",");
    candidate.push_str(&format!(
        "{{\"type\":\"tokens\",\"ids\":[{zeros}],\"fed_ids\":[{ones}]}}\n"
    ));
    let reference = temp_file("reference-fed", &reference);
    let candidate = temp_file("candidate-fed", &candidate);
    let output = run(&reference, &candidate);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("teacher-forced token streams differ"));
}

#[test]
fn comparator_rejects_nonnumeric_full_logits() {
    let bad = "{\"type\":\"logits\",\"step\":0,\"argmax\":0,\"margin\":1.0,\"finite\":1,\"values\":[null]}\n{\"type\":\"tokens\",\"ids\":[0]}\n";
    let reference = temp_file("reference-null-value", bad);
    let candidate = temp_file("candidate-null-value", bad);
    let output = run_with_args(&reference, &candidate, false);
    fs::remove_file(reference).unwrap();
    fs::remove_file(candidate).unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("is not numeric"));
}
