import os
import sys
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --- THE PICKLE FIX ---
# Tell Python to look inside the 'utils' folder right next to this file
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
# ----------------------

class VolleyballDataset(Dataset):
    def __init__(
        self,
        data_root,
        pkl_file,
        split_video_ids,
        video_dir_name="sample",
        transform=None,
    ):
        self.data_root = data_root
        self.video_dir_name = video_dir_name
        self.transform = transform

        print(f"Loading annotations from {pkl_file}...")
        with open(pkl_file, "rb") as f:
            self.annotations = pickle.load(f)

        # 1. Flatten the nested dictionary into a simple list so PyTorch can index it easily.
        self.clip_list = []
        for vid in split_video_ids:
            # --- TYPE MISMATCH FIX ---
            # Check if the video ID exists as a string OR an integer in the PKL file
            clip_dict = None
            if vid in self.annotations:
                clip_dict = self.annotations[vid]
            elif vid.isdigit() and int(vid) in self.annotations:
                clip_dict = self.annotations[int(vid)]

            if clip_dict is not None:
                for clip_id, clip_data in clip_dict.items():
                    self.clip_list.append(
                        {
                            "video_id": str(vid),     # Force to string for file paths
                            "clip_id": str(clip_id),  # Force to string for file paths
                            "category": clip_data["category"],
                            "frames": list(clip_data["frame_boxes_dct"].keys()),
                        }
                    )
        
        # Debugging print to help us if it is still 0
        if len(self.clip_list) == 0:
            print(f"⚠️ DEBUG: PKL dictionary keys look like: {list(self.annotations.keys())[:5]}")

        print(f"Loaded {len(self.clip_list)} clips for this dataset split.")

        # 2. Map the exact string labels from the text files to integers (0-7)
        self.category_to_idx = {
            "l-pass": 0,
            "r-pass": 1,
            "l-spike": 2,
            "r_spike": 3,
            "l_set": 4,
            "r_set": 5,
            "l_winpoint": 6,
            "r_winpoint": 7,
        }

    def __len__(self):
        return len(self.clip_list)

    def _clean_label(self, raw_label):
        label = (
            raw_label.strip()
            .replace(" ", "_")
            .replace("Left", "l")
            .replace("Right", "r")
            .lower()
        )

        if label == "l_pass": return "l-pass"
        if label == "r_pass": return "r-pass"
        if label == "l_spike": return "l-spike"

        return label

    def __getitem__(self, idx):
        clip_info = self.clip_list[idx]

        # Find the middle frame of the sequence
        frames = sorted(clip_info["frames"])
        middle_frame_id = frames[len(frames) // 2]

        # Construct the absolute path to the JPG
        img_path = os.path.join(
            self.data_root,
            self.video_dir_name,
            clip_info["video_id"],
            clip_info["clip_id"],
            f"{middle_frame_id}.jpg",
        )

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        clean_label_str = self._clean_label(clip_info["category"])
        label_idx = self.category_to_idx[clean_label_str]

        return image, label_idx
