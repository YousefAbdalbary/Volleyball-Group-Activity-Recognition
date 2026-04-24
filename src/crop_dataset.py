import os
import sys
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Tell Python to look inside the 'utils' folder for BoxInfo
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


class PlayerCropDataset(Dataset):
    def __init__(
        self,
        data_root,
        pkl_file,
        split_video_ids,
        video_dir_name="videos",
        transform=None,
    ):
        """
        Dataset for Baseline B3 (Phase A):
        Extracts individual player crops and their 9-class action labels.
        """
        self.data_root = data_root
        self.video_dir_name = video_dir_name
        self.transform = transform

        print(f"Loading annotations from {pkl_file}...")
        with open(pkl_file, "rb") as f:
            self.annotations = pickle.load(f)

        # 1. Map the 9 individual person actions to integers
        self.person_categories = [
            "waiting",
            "setting",
            "digging",
            "falling",
            "spiking",
            "blocking",
            "jumping",
            "moving",
            "standing",
        ]
        self.category_to_idx = {
            cat: idx for idx, cat in enumerate(self.person_categories)
        }

        # 2. Flatten the data into individual player crops
        self.crops_list = []
        for vid in split_video_ids:
            # Handle Kaggle's string vs integer quirks
            clip_dict = None
            if vid in self.annotations:
                clip_dict = self.annotations[vid]
            elif str(vid) in self.annotations:
                clip_dict = self.annotations[str(vid)]
            elif str(vid).isdigit() and int(vid) in self.annotations:
                clip_dict = self.annotations[int(vid)]

            if clip_dict is not None:
                for clip_id, clip_data in clip_dict.items():
                    # For Phase A, we only extract crops from the MIDDLE frame of each clip
                    frames = sorted(list(clip_data["frame_boxes_dct"].keys()))
                    middle_frame_id = frames[len(frames) // 2]

                    # Get the list of players (BoxInfo objects) in this specific frame
                    players = clip_data["frame_boxes_dct"][middle_frame_id]

                    for player in players:
                        self.crops_list.append(
                            {
                                "video_id": str(vid),
                                "clip_id": str(clip_id),
                                "frame_id": str(middle_frame_id),
                                "box": player.box,  # [x1, y1, x2, y2]
                                "category": player.category.lower(),
                            }
                        )

        print(f"Loaded {len(self.crops_list)} individual player crops for this split.")

    def __len__(self):
        return len(self.crops_list)

    def __getitem__(self, idx):
        crop_info = self.crops_list[idx]

        # Construct the absolute path to the full JPG image
        img_path = os.path.join(
            self.data_root,
            self.video_dir_name,
            crop_info["video_id"],
            crop_info["clip_id"],
            f"{crop_info['frame_id']}.jpg",
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")

        # Load the full image
        full_image = Image.open(img_path).convert("RGB")
        width, height = full_image.size

        # 3. Apply the Bounding Box Crop
        x1, y1, x2, y2 = crop_info["box"]

        # Safety check: Ensure coordinates don't go outside the image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # PIL uses (left, upper, right, lower)
        crop_image = full_image.crop((x1, y1, x2, y2))

        # Apply transformations (Resize to 224x224, Normalize, ToTensor)
        if self.transform:
            crop_image = self.transform(crop_image)

        # Get the label index
        label_str = crop_info["category"]
        label_idx = self.category_to_idx.get(
            label_str, 0
        )  # Defaults to 'waiting' if weird label

        return crop_image, label_idx
