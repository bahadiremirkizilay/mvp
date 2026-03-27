"""
Validate self-collected data for lie detection pipeline readiness.

Checks per subject folder:
- video.mp4 exists
- labels.json exists
- labels.json schema consistency
- deception-labeled segment counts

Usage:
    python fusion/validate_self_collected_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def validate_labels_schema(path: Path) -> Dict:
    issues: List[str] = []
    deception_segments = 0
    truth_segments = 0
    valid_segments = 0

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "valid": False,
            "issues": [f"Invalid JSON: {exc}"],
            "deception_segments": 0,
            "truth_segments": 0,
            "valid_segments": 0,
        }

    required_top = ["subject_id", "session_id", "conditions"]
    for key in required_top:
        if key not in payload:
            issues.append(f"Missing top-level key: {key}")

    sampling = payload.get("sampling", {})
    audio_included = bool(sampling.get("audio_included", False))
    audio_filename = sampling.get("audio_filename")

    if audio_included and not audio_filename:
        issues.append("sampling.audio_filename must be set when audio_included=true")

    conditions = payload.get("conditions", [])
    if not isinstance(conditions, list) or len(conditions) == 0:
        issues.append("conditions must be a non-empty list")
        return {
            "valid": False,
            "issues": issues,
            "deception_segments": 0,
            "truth_segments": 0,
            "valid_segments": 0,
        }

    for i, seg in enumerate(conditions):
        for key in ["segment_id", "start_sec", "end_sec", "task_type", "deception_label"]:
            if key not in seg:
                issues.append(f"Segment[{i}] missing key: {key}")

        s = float(seg.get("start_sec", -1))
        e = float(seg.get("end_sec", -1))
        if e <= s:
            issues.append(f"Segment[{i}] has invalid time range: start={s}, end={e}")
            continue

        label = seg.get("deception_label")
        if label in ["deception", "truth"]:
            valid_segments += 1
            if label == "deception":
                deception_segments += 1
            if label == "truth":
                truth_segments += 1

    if deception_segments == 0:
        issues.append("No deception-labeled segments found")
    if truth_segments == 0:
        issues.append("No truth-labeled segments found")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "deception_segments": deception_segments,
        "truth_segments": truth_segments,
        "valid_segments": valid_segments,
        "audio_included": audio_included,
        "audio_filename": audio_filename,
    }


def main() -> None:
    root = Path("data/self_collected")
    subjects = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("subject_")])

    report = {
        "root": str(root),
        "subject_count": len(subjects),
        "subjects": [],
        "summary": {
            "ready_subjects": 0,
            "total_deception_segments": 0,
            "total_truth_segments": 0,
            "blocking_issues": [],
        },
    }

    if len(subjects) == 0:
        report["summary"]["blocking_issues"].append(
            "No subject_* folders found in data/self_collected"
        )

    for subject_dir in subjects:
        video_path = subject_dir / "video.mp4"
        labels_path = subject_dir / "labels.json"

        subject_info = {
            "subject": subject_dir.name,
            "video_exists": video_path.exists(),
            "labels_exists": labels_path.exists(),
            "audio_exists": None,
            "schema_valid": False,
            "issues": [],
            "deception_segments": 0,
            "truth_segments": 0,
        }

        if not video_path.exists():
            subject_info["issues"].append("Missing video.mp4")
        if not labels_path.exists():
            subject_info["issues"].append("Missing labels.json")

        if labels_path.exists():
            schema = validate_labels_schema(labels_path)
            subject_info["schema_valid"] = schema["valid"]
            subject_info["issues"].extend(schema["issues"])
            subject_info["deception_segments"] = schema["deception_segments"]
            subject_info["truth_segments"] = schema["truth_segments"]
            if schema.get("audio_included"):
                audio_path = subject_dir / schema.get("audio_filename")
                subject_info["audio_exists"] = audio_path.exists()
                if not audio_path.exists():
                    subject_info["issues"].append(f"Missing audio file: {audio_path.name}")

        if (
            subject_info["video_exists"]
            and subject_info["labels_exists"]
            and subject_info["schema_valid"]
        ):
            report["summary"]["ready_subjects"] += 1

        report["summary"]["total_deception_segments"] += subject_info["deception_segments"]
        report["summary"]["total_truth_segments"] += subject_info["truth_segments"]

        report["subjects"].append(subject_info)

    # Global readiness thresholds for starting true lie-detection training
    if report["summary"]["ready_subjects"] < 12:
        report["summary"]["blocking_issues"].append(
            "Need at least 12 ready subjects for first LOSO deception experiment"
        )
    if report["summary"]["total_deception_segments"] < 120:
        report["summary"]["blocking_issues"].append(
            "Need at least 120 deception segments"
        )
    if report["summary"]["total_truth_segments"] < 120:
        report["summary"]["blocking_issues"].append(
            "Need at least 120 truth segments"
        )

    out_path = Path("checkpoints/fusion/self_collected_readiness.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 80)
    print("SELF-COLLECTED DATA READINESS REPORT")
    print("=" * 80)
    print(f"Subjects found: {report['subject_count']}")
    print(f"Ready subjects: {report['summary']['ready_subjects']}")
    print(f"Truth segments: {report['summary']['total_truth_segments']}")
    print(f"Deception segments: {report['summary']['total_deception_segments']}")

    if report["summary"]["blocking_issues"]:
        print("\nBlocking issues:")
        for issue in report["summary"]["blocking_issues"]:
            print(f"  - {issue}")
    else:
        print("\nDataset is READY for first LOSO deception training.")

    print(f"\nDetailed report: {out_path}")


if __name__ == "__main__":
    main()
