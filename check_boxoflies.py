import os

os.chdir(r'c:\Users\furka\Desktop\projects\mvp\data\boxoflies')

videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
audios = [f for f in os.listdir('.') if f.endswith('.wav')]

print(f'Total MP4 videos: {len(videos)}')
print(f'Total WAV files: {len(audios)}')

# Extract unique subjects
subjects = set()
for f in videos:
    parts = f.split('_')
    if len(parts) >= 2:
        subjects.add(parts[1])  # The name

print(f'Unique subjects: {sorted(subjects)}')

# Parse labels
labels = {}
with open('lie_detection_wav.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            wav_file = parts[0]
            gender = parts[1]
            is_truth = parts[2] == 'true'
            labels[wav_file] = (gender, is_truth)

true_count = sum(1 for v in labels.values() if v[1])
lie_count = len(labels) - true_count
print(f'\nLabeled segments: {len(labels)}')
print(f'  Truth: {true_count}, Lies: {lie_count}')
print(f'Subjects count: {len(subjects)}')
print(f'Label distribution by subject:')
for subject in sorted(subjects):
    subject_videos = [f for f in videos if f'_{subject}.mp4' in f]
    subject_lies = sum(1 for f in subject_videos for lbl in [labels.get(f[:-4]+'_audio.wav')] if lbl and not lbl[1])
    subject_truths = sum(1 for f in subject_videos for lbl in [labels.get(f[:-4]+'_audio.wav')] if lbl and lbl[1])
    print(f'  {subject}: {len(subject_videos)} videos')
