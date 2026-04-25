import os
import sys
import pickle
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# Import your trained Scout model
from src.models.baseline_b3 import BaselineB3PersonModel

# Tell Python where to find BoxInfo
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "utils"))

# --- 1. CONFIGURATION ---
if os.path.exists("/kaggle/working"):
    DATA_ROOT = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball"
    OUTPUT_ROOT = "/kaggle/working/features"
    PKL_FILE = os.path.join(DATA_ROOT, "annot_all.pkl")
    MODEL_WEIGHTS = "/kaggle/working/baseline_b3_person_model.pth"

    if os.path.exists(os.path.join(DATA_ROOT, "videos", "videos")):
        VIDEOS_DIR_NAME = "videos/videos"
    else:
        VIDEOS_DIR_NAME = "videos"
else:
    DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
    OUTPUT_ROOT = os.path.join(DATA_ROOT, "features")
    PKL_FILE = os.path.join(DATA_ROOT, "annotations", "annot_all.pkl")
    MODEL_WEIGHTS = os.path.join(DATA_ROOT, "baseline_b3_person_model.pth")
    VIDEOS_DIR_NAME = "sample"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extracting features using device: {device}")

    # Create the output directories if they don't exist
    os.makedirs(os.path.join(OUTPUT_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "val"), exist_ok=True)

    # --- 2. LOAD THE HEADLESS SCOUT MODEL ---
    print("Loading Scout Model...")
    model = BaselineB3PersonModel(num_classes=9, fine_tune_all=False)

    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(
            torch.load(MODEL_WEIGHTS, map_location=device, weights_only=True)
        )
    else:
        print(f"⚠️ WARNING: Could not find trained weights at {MODEL_WEIGHTS}.")

    model.model.fc = nn.Identity()
    model = model.to(device)
    model.eval()

    # --- 3. PREPARE THE DATA ---
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    with open(PKL_FILE, "rb") as f:
        annotations = pickle.load(f)

    TRAIN_IDS = [
        "1",
        "3",
        "6",
        "7",
        "10",
        "13",
        "15",
        "16",
        "18",
        "22",
        "23",
        "31",
        "32",
        "36",
        "38",
        "39",
        "40",
        "41",
        "42",
        "48",
        "50",
        "52",
        "53",
        "54",
    ]
    VAL_IDS = [
        "0",
        "2",
        "8",
        "12",
        "17",
        "19",
        "24",
        "26",
        "27",
        "28",
        "30",
        "33",
        "46",
        "49",
        "51",
    ]

    available_videos_path = os.path.join(DATA_ROOT, VIDEOS_DIR_NAME)
    available_videos = [
        d
        for d in os.listdir(available_videos_path)
        if os.path.isdir(os.path.join(available_videos_path, d))
    ]

    # --- 4. THE EXTRACTION LOOP ---
    with torch.no_grad():
        for vid in tqdm(available_videos, desc="Extracting Features"):

            split_folder = (
                "train" if vid in TRAIN_IDS else "val" if vid in VAL_IDS else None
            )
            if not split_folder:
                continue

            # BULLETPROOF DICTIONARY LOOKUP
            clip_dict = None
            if vid in annotations:
                clip_dict = annotations[vid]
            elif str(vid) in annotations:
                clip_dict = annotations[str(vid)]
            elif str(vid).isdigit() and int(vid) in annotations:
                clip_dict = annotations[int(vid)]

            if not clip_dict:
                continue

            for clip_id, clip_data in clip_dict.items():

                frames = sorted(list(clip_data["frame_boxes_dct"].keys()))
                middle_frame_id = frames[len(frames) // 2]
                players = clip_data["frame_boxes_dct"][middle_frame_id]

                img_path = os.path.join(
                    available_videos_path,
                    str(vid),
                    str(clip_id),
                    f"{middle_frame_id}.jpg",
                )

                # If image is missing, print a loud warning instead of skipping silently
                if not os.path.exists(img_path):
                    print(f"⚠️ MISSING IMAGE: {img_path}")
                    continue

                full_image = Image.open(img_path).convert("RGB")
                width, height = full_image.size

                player_tensors = []
                for player in players:
                    x1, y1, x2, y2 = player.box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    crop_img = full_image.crop((x1, y1, x2, y2))
                    crop_tensor = transform(crop_img)
                    player_tensors.append(crop_tensor)

                if len(player_tensors) == 0:
                    continue

                batch_tensor = torch.stack(player_tensors).to(device)
                features = model(batch_tensor)
                team_summary, _ = torch.max(features, dim=0)

                save_name = f"{vid}_{clip_id}.pt"
                save_path = os.path.join(OUTPUT_ROOT, split_folder, save_name)

                torch.save(team_summary.cpu(), save_path)

    print("\nFeature extraction complete! All files saved to", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
