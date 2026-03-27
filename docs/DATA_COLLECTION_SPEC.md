# Self-Collected Data Spec for Lie Detection

This document defines exactly what to add when new data is needed.

For dataset prioritization (what is mandatory vs optional), see:
- [docs/REQUIRED_DATASETS.md](docs/REQUIRED_DATASETS.md)

## Low Storage Workaround (Important)

If your C disk is full, do not copy large datasets into the project folder.
Use an external drive path and create a junction link.

Example:

python scripts/setup/link_external_dataset.py --external "D:/datasets/BagOfLies" --target "data/BagOfLies"

This makes `data/BagOfLies` appear inside the project without consuming extra C disk space.

## Folder Structure

- data/self_collected/subject_001/video.mp4
- data/self_collected/subject_001/audio.wav (optional but recommended)
- data/self_collected/subject_001/labels.json
- data/self_collected/subject_001/ecg_reference.csv (optional)

Use [data/self_collected/labels_template.json](../data/self_collected/labels_template.json) as template.

## Minimum Dataset for First Real LOSO Deception Training

- At least 12 subjects
- At least 120 truth segments
- At least 120 deception segments
- Each segment should be 15-40 seconds
- Face visibility should be mostly stable

## Label Rules

- deception_label values:
  - truth
  - deception
  - null (for baseline/non-scored segments)
- Every scored segment must have:
  - start_sec
  - end_sec
  - task_type
  - prompt

## Quality Recommendations

- Lighting: stable indoor light
- Camera: fixed position, frontal face
- Distance: ~50-80 cm
- Motion: low to medium (avoid large head turns)
- FPS: 30 preferred

## Audio Recommendations

- Record a separate mono WAV file when possible
- Preferred sample rate: 16000 Hz or 22050 Hz
- Keep microphone distance stable
- Reduce room echo and background TV/music
- Avoid clipping; normal conversation level is enough
- Keep question-answer protocol time-aligned with video

Audio is not mandatory for starting the first lie-risk proxy, but it is recommended for the later deception model.

Useful vocal cues we can add:
- short-term energy
- speaking rate proxy
- pitch proxy / pitch instability
- zero-crossing rate
- spectral centroid / bandwidth
- voiced ratio / pause structure

## Readiness Check Command

Run:

python fusion/validate_self_collected_data.py

Report path:

checkpoints/fusion/self_collected_readiness.json

When report has no blocking issues, we can start true deception training.

## External Dataset Config (Optional)

Use local config template:

- [config/datasets.local.example.yaml](../config/datasets.local.example.yaml)

Create your own:

- config/datasets.local.yaml

and fill external absolute paths there.
