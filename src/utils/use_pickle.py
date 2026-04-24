# create_dummy_labels.py
import os

DATA_ROOT = (
    "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
)
SAMPLE_ROOT = os.path.join(DATA_ROOT, "sample")

# For each video, create annotations.txt with dummy labels
# Using actual 8 classes from the paper
CLASSES = [
    "r_set",
    "r_spike",
    "r_pass",
    "r_winpoint",
    "l_winpoint",
    "l_pass",
    "l_spike",
    "l_set",
]

for video_dir in sorted(os.listdir(SAMPLE_ROOT)):
    video_path = os.path.join(SAMPLE_ROOT, video_dir)
    if not os.path.isdir(video_path):
        continue

    annot_file = os.path.join(video_path, "annotations.txt")

    # Get actual clip folders
    clips = sorted(
        [
            d
            for d in os.listdir(video_path)
            if os.path.isdir(os.path.join(video_path, d))
        ]
    )

    print(f"\n🎬 Video {video_dir}: found {len(clips)} clips")

    # Write dummy labels (cycle through classes)
    with open(annot_file, "w") as f:
        for i, clip in enumerate(clips):
            label = CLASSES[i % len(CLASSES)]
            line = f"{clip}.jpg {label}\n"
            f.write(line)
            print(f"   {clip}.jpg → {label}")

    print(f"   ✅ Created: {annot_file}")

print("\n🎉 Done! Now re-run your loader.")
