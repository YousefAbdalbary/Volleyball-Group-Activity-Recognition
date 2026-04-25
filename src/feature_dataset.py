import os
import pickle
import torch
from torch.utils.data import Dataset
import sys

# --- THE PICKLE FIX ---
# Tell Python to look inside the 'utils' folder right next to this file
# so it knows how to unpack the BoxInfo objects!
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


class FeatureDataset(Dataset):
    """
    Phase C Dataset:
    Loads the pre-extracted 2048-dimensional Team Summaries (.pt files)
    and pairs them with the 8-class Group Activity labels.
    """

    def __init__(self, features_dir, pkl_file):
        self.features_dir = features_dir

        with open(pkl_file, "rb") as f:
            self.annotations = pickle.load(f)

        # 1. Map the 8 group activities to integers
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

        # 2. Find all the .pt files in the directory
        self.pt_files = [f for f in os.listdir(features_dir) if f.endswith(".pt")]

        """
        [CONCEPT CHECK: HOW MANY FILES?]
        For a single clip (like 93635), there are 12 players.
        However, in extract_features.py, we used torch.max() to smash all 12 
        players together into ONE Team Summary. 
        Therefore, we DO NOT have 12 files per clip. 
        We only have exactly 1 file per clip! (e.g., "7_93635.pt")
        """

        self.data_list = []

        for pt_file in self.pt_files:
            # The file name format is "vid_clipid.pt" (e.g., "7_93635.pt")
            vid_str, clip_str = pt_file.replace(".pt", "").split("_")

            # Find the group label in the PKL dictionary safely
            clip_dict = self.annotations.get(vid_str) or self.annotations.get(
                int(vid_str)
            )

            if clip_dict:
                # Check if the key exists as a string OR as an integer
                if clip_str in clip_dict:
                    raw_label = clip_dict[clip_str]["category"]
                elif int(clip_str) in clip_dict:
                    raw_label = clip_dict[int(clip_str)]["category"]
                else:
                    continue  # Could not find the clip in annotations

                # ⚠️ Notice: We do NOT redeclare raw_label here anymore.
                # It was safely captured by the if/elif block above!

                clean_label = self._clean_label(raw_label)
                label_idx = self.category_to_idx[clean_label]

                self.data_list.append(
                    {
                        "file_path": os.path.join(features_dir, pt_file),
                        "label": label_idx,
                    }
                )

        print(f"Loaded {len(self.data_list)} extracted features from {features_dir}")

    def _clean_label(self, raw_label):
        # Cleans up messy text from the dataset (e.g., "Left Spike" -> "l-spike")

        # 1. Bulletproof: .lower() happens FIRST
        label = (
            raw_label.lower()
            .strip()
            .replace(" ", "_")
            .replace("left", "l")
            .replace("right", "r")
        )

        # 2. The Hyphen Hacks (Because your dictionary uses hyphens for these three!)
        if label == "l_pass":
            return "l-pass"
        if label == "r_pass":
            return "r-pass"
        if label == "l_spike":
            return "l-spike"

        return label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # Load the 2048 tensor directly from the hard drive
        feature_tensor = torch.load(item["file_path"], weights_only=True)
        label = item["label"]

        return feature_tensor, label
