import os
import sys
import pickle
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm  # This creates a nice progress bar!

# Import your trained Scout model
from src.models.baseline_b3 import BaselineB3PersonModel

# Tell Python where to find BoxInfo
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "utils"))


# --- 1. CONFIGURATION ---
if os.path.exists("/kaggle/working"):
    DATA_ROOT = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball"
    OUTPUT_ROOT = "/kaggle/working/features"  # We will save all .pt files in a new 'features' folder
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
    print(f"extracting features using device: {device}")

    # Create the output directories if they don't exist
    os.makedirs(os.path.join(OUTPUT_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "val"), exist_ok=True)

    # --- 2. LOAD THE HEADLESS SCOUT MODEL ---
    print("Loading Scout Model...")
    model = BaselineB3PersonModel(num_classes=9, fine_tune_all=False)

    # Load the weights we just trained!
    if os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    else:
        print(
            f"⚠️ WARNING: Could not find trained weights at {MODEL_WEIGHTS}. Using untrained ResNet!"
        )

    # THE CHOPPING BLOCK:
    # We replace the final Linear layer (which outputs 9 classes)
    # with nn.Identity(), which does absolutely nothing.
    # Now, the model stops at the 2048 features!
    """
    i remove the last layer of the model that classify the 9 classes , and replace it with identity layer, 
    so the result will be the features of the image 2048 features
    i do this because i want to use the features of the image to train the group model
    """
    model.model.fc = nn.Identity()

    model = model.to(device)
    # i am not learing , i use the weights from the previous training
    # so i will not use the optimizer
    model.eval()  # Lock the model into evaluation mode

    # --- 3. PREPARE THE IMAGE TRANSFORMER ---
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load annotations
    with open(PKL_FILE, "rb") as f:
        annotations = pickle.load(f)

    # Official Splits
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
    with torch.no_grad():  # Tell PyTorch not to track gradients to save massive amounts of RAM
        for vid in tqdm(available_videos, desc="Extracting Features"):
            # Determine if this video goes in the train or val folder
            split_folder = (
                "train" if vid in TRAIN_IDS else "val" if vid in VAL_IDS else None
            )
            if not split_folder:
                continue
            clip_dict = annotations.get(vid) or annotations.get(int(vid))
            if not clip_dict:
                continue

                for clip_id, clip_data in clip_dict.items():

                    # find the middle frame of the clip
                    frames = sorted(list(clip_data["frame_boxes_dct"].keys()))
                    middle_frame_id = frames[len(frames) // 2]
                    players = clip_data["frame_boxes_dct"][middle_frame_id]

                    # Load the full image
                img_path = os.path.join(
                    available_videos_path,
                    str(vid),
                    str(clip_id),
                    f"{middle_frame_id}.jpg",
                )
                if not os.path.exists(img_path):
                    continue

                full_image = Image.open(img_path).convert("RGB")
                width, height = full_image.size

                # Cut out all 12 players and transform them
                player_tensors = []
                for player in players:
                    x1, y1, x2, y2 = player.box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    crop_img = full_image.crop((x1, y1, x2, y2))
                    crop_tensor = transform(crop_img)
                    player_tensors.append(crop_tensor)

                # If no players were found, skip this clip
                if len(player_tensors) == 0:
                    continue

                # Stack the 12 individual tensors into one batch: shape becomes [12, 3, 224, 224]
                batch_tensor = torch.stack(player_tensors).to(device)

                # Pass the batch of 12 through the headless Scout model
                # Output shape will be [12, 2048]
                """
                as the batch size is 12 , the output will be [12, 2048]
                beacuse i have replace the last fc layer with identity layer
                so i get the features of the image as 2048 features for each image
                and i will save the features of the 12 players in a file
                the file will be like this: 
                {
                    "vid": {
                        "clip_id": {
                            "player_id": {
                                "features": [feature1, feature2, ..., feature2048],
                                "label": label
                            }
                        }
                    }
                }
                """
                features = model(batch_tensor)

                # THE MAGIC STEP: Max Pooling
                # We tell PyTorch to look across dimension 0 (the 12 players)
                # and find the maximum value for each of the 2048 features.
                # The output shape collapses to a Team Summary of [2048]
                """
                the _ return the index of the max value 
                and team_summary return the max value
                features shape is [12, 2048]
                so the max value will be the max value of each feature across the 12 players
                the dim of result will be [2048] or 1 * 2048
                    dim 0 is the 12 players
                    dim 1 is the 2048 features

                """
                team_summary, _ = torch.max(features, dim=0)

                # Save the tensor to the hard drive
                # We name it "vid_clip.pt" (e.g., "7_93635.pt")
                """
                I will save the features of the 12 players in a file
                the file will be like this: 
                {
                    "vid": {
                        "clip_id": {
                            "player_id": {
                                "features": [feature1, feature2, ..., feature2048],
                                "label": label
                            }
                        }
                    }
                }

                """
                save_name = f"{vid}_{clip_id}.pt"
                save_path = os.path.join(OUTPUT_ROOT, split_folder, save_name)

                # Move back to CPU before saving to avoid VRAM leaks
                """ 
                vram leak mean that the memory is not released after the operation
                so i move the tensor to the cpu before saving it
                how moving to cpu save it ?? 
                it mean that the tensor is not released from the gpu memory
                and it will be released after the program is terminated
                so i move the tensor to the cpu before saving it
                """
                torch.save(team_summary.cpu(), save_path)

    print("\nFeature extraction complete! All files saved to", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
