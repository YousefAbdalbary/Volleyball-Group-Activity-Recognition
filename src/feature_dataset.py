import os
import pickle
import torch
from torch.utils.data import Dataset
import sys

# Add utils folder to Python path so we can import BoxInfo if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))


class FeatureDataset(Dataset):
    """
    Phase C Dataset: Group Activity Classifier on Pre-Extracted Features

    WHAT THIS DOES:
    Loads .pt files containing 2048-D team feature vectors and pairs them
    with group activity labels (8 classes) for training a simple classifier.

    WHAT THIS IS NOT:
    - NOT processing raw images (that's done in PlayerCropDataset)
    - NOT handling individual players (features are already max-pooled)
    - NOT temporal (each sample is one frame/clip, not a sequence)

    INPUT:  .pt files from extract_features_for_b3.py (team summaries)
    OUTPUT: (feature_tensor(2048,), label(int)) pairs for DataLoader

    EXAMPLE FLOW:
    File "7_93635.pt" → load tensor([0.2, -0.5, 1.1, ...]) shape (2048,)
    Label from annot_all.pkl for video 7, clip 93635 → "r_pass" → index 1
    Returns: (tensor(2048,), 1)
    """

    def __init__(self, features_dir, pkl_file):
        """
        Build the dataset by matching .pt feature files to their labels.

        Args:
            features_dir: Path to folder containing .pt files
                Example: "/project/data/features/train"
                Contains: ["7_93635.pt", "7_93636.pt", ...]

            pkl_file: Path to annotation pickle
                Example: "/project/data/annotations/annot_all.pkl"
                Contains nested dict: {video_id: {clip_id: {"category": "r_pass", ...}}}
        """
        self.features_dir = features_dir

        # Load the annotation dictionary created by volleyball_annot_loader.py
        # Structure: {
        #     "7": {                           # video_id as string
        #         "93635": {                   # clip_id
        #             "category": "r_pass",    # GROUP activity (8 classes)
        #             "frame_boxes_dct": {...} # player boxes (not used here)
        #         },
        #         "93636": {...},
        #     },
        #     "10": {...},
        # }
        with open(pkl_file, "rb") as f:
            self.annotations = pickle.load(f)

        # Map 8 group activity strings to integer indices for CrossEntropyLoss
        # CrossEntropyLoss expects labels as integers 0-7, not strings
        # FIXED: All standardized to underscore format for consistency
        self.category_to_idx = {
            "l_pass": 0,  # left team passing
            "r_pass": 1,  # right team passing
            "l_spike": 2,  # left team spiking
            "r_spike": 3,  # right team spiking
            "l_set": 4,  # left team setting
            "r_set": 5,  # right team setting
            "l_winpoint": 6,  # left team wins point
            "r_winpoint": 7,  # right team wins point
        }

        # Get all .pt files in the features directory
        # Each .pt file = ONE clip's team summary (already max-pooled across 12 players)
        #
        # Example listing:
        # features_dir = "/project/data/features/train"
        # pt_files = ["7_93635.pt", "7_93636.pt", "10_18360.pt", ...]
        #
        # Naming convention: "{video_id}_{clip_id}.pt"
        # Example: "7_93635.pt" → video 7, clip 93635
        self.pt_files = [f for f in os.listdir(features_dir) if f.endswith(".pt")]

        # Build list of (file_path, label) pairs for __getitem__
        # Each entry = one training sample
        self.data_list = []

        for pt_file in self.pt_files:
            # Parse filename to get video_id and clip_id
            #
            # EXAMPLE:
            # pt_file = "7_93635.pt"
            # base_name = "7_93635"  (after removing .pt)
            # vid_str = "7"          (video folder name)
            # clip_str = "93635"     (clip folder name)
            #
            # FIXED: Use rsplit("_", 1) instead of split("_")
            # WHY: Video IDs like "10" won't break, and clip IDs with underscores
            #      (unlikely but possible) are handled correctly
            #
            # BAD:  "10_18360".split("_") → ["10", "18360"] ✓ works
            #       "10_18360_extra".split("_") → ["10", "18360", "extra"] ✗ breaks!
            # GOOD: "10_18360_extra".rsplit("_", 1) → ["10_18360", "extra"] ✓ safer
            base_name = pt_file.replace(".pt", "")
            vid_str, clip_str = base_name.rsplit("_", 1)

            # Lookup video in annotations dictionary
            # Try string key first, then integer key (handles pickle inconsistencies)
            #
            # EXAMPLE:
            # annotations = {"7": {...}, 10: {...}}  (mixed key types possible)
            # vid_str = "7"
            # annotations.get("7") → returns dict ✓
            # annotations.get(7) → returns None (fallback not needed)
            clip_dict = self.annotations.get(vid_str) or self.annotations.get(
                int(vid_str)
            )

            # Skip if video not found in annotations
            if not clip_dict:
                print(f"Warning: Video {vid_str} not found in annotations")
                continue

            # FIXED: Single assignment with proper existence check
            # Get the group activity label for this specific clip
            #
            # EXAMPLE lookup:
            # clip_dict = {
            #     "93635": {"category": "r_pass", ...},
            #     "93636": {"category": "l_set", ...},
            # }
            # clip_str = "93635"
            # clip_dict["93635"] → {"category": "r_pass", ...}
            # raw_label = "r_pass"
            if clip_str in clip_dict:
                raw_label = clip_dict[clip_str]["category"]
            elif int(clip_str) in clip_dict:
                # Fallback: clip_id might be stored as integer in pickle
                raw_label = clip_dict[int(clip_str)]["category"]
            else:
                print(f"Warning: Clip {clip_str} not found in video {vid_str}")
                continue  # Skip this .pt file — no label available

            # Clean the raw label text to match our dictionary keys
            #
            # EXAMPLE transformation:
            # raw_label = "Right Spike" (from annotation file)
            # Step 1: lower() → "right spike"
            # Step 2: strip() → "right spike" (no change)
            # Step 3: replace(" ", "_") → "right_spike"
            # Step 4: replace("-", "_") → "right_spike" (no hyphen)
            # Step 5: replace("right", "r") → "r_spike"
            # Result: "r_spike" ✓ matches dictionary key
            clean_label = self._clean_label(raw_label)

            # Map cleaned string label to integer index
            #
            # EXAMPLE:
            # clean_label = "r_spike"
            # category_to_idx["r_spike"] → 3
            # label_idx = 3
            label_idx = self.category_to_idx.get(clean_label)

            # Skip if label not in our known classes
            if label_idx is None:
                print(f"Warning: Unknown label '{clean_label}' from raw '{raw_label}'")
                continue

            # Store the file path and label for this sample
            #
            # EXAMPLE entry:
            # {
            #     "file_path": "/project/data/features/train/7_93635.pt",
            #     "label": 1,
            # }
            self.data_list.append(
                {
                    "file_path": os.path.join(features_dir, pt_file),
                    "label": label_idx,
                }
            )

        # Final count
        # EXAMPLE: "Loaded 2450 samples from /project/data/features/train"
        print(f"Loaded {len(self.data_list)} samples from {features_dir}")

    def _clean_label(self, raw_label):
        """
        Standardize label text from annotation files to match dictionary keys.

        Dataset labels come in various formats:
        - "Right Spike" (title case, space separator)
        - "left-pass" (lowercase, hyphen separator)
        - "r_set" (already clean)
        - "L Winpoint" (mixed case, space)

        We convert all to: lowercase, underscore separator, l/r prefix

        EXAMPLES:
        "Left Spike"     → "l_spike"
        "right pass"     → "r_pass"
        "r-set"          → "r_set"
        "L Winpoint"     → "l_winpoint"
        "r_pass"         → "r_pass" (already clean, no change)
        """
        label = (
            raw_label.lower()  # "Right Spike" → "right spike"
            .strip()  # Remove leading/trailing whitespace
            .replace(" ", "_")  # "right spike" → "right_spike"
            .replace("-", "_")  # "right-spike" → "right_spike" (standardize)
            .replace("left", "l")  # "left_spike" → "l_spike"
            .replace("right", "r")  # "right_spike" → "r_spike"
        )
        return label

    def __len__(self):
        """
        Return total number of valid (feature, label) pairs.

        EXAMPLE:
        len(dataset) → 2450
        (meaning we have 2450 .pt files with matching annotations)
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load and return a single sample.

        Args:
            idx: Integer index (0 to len(dataset)-1)

        Returns:
            feature_tensor: torch.Tensor of shape (2048,)
                Contains the pre-extracted team representation.
                Values are float32, can be positive or negative.
                EXAMPLE: tensor([ 0.23, -1.56,  0.89, ...,  2.11])

            label: int (0-7)
                Integer class index for CrossEntropyLoss.
                EXAMPLE: 1 (which means "r_pass")

        FULL EXAMPLE:
        idx = 0
        item = {
            "file_path": "/project/data/features/train/7_93635.pt",
            "label": 1,
        }

        feature_tensor = torch.load("/project/data/features/train/7_93635.pt")
        # → tensor([0.23, -1.56, 0.89, ..., 2.11]) shape (2048,)

        label = 1

        return (tensor([0.23, -1.56, 0.89, ..., 2.11]), 1)
        """
        item = self.data_list[idx]

        # Load the pre-computed 2048-D team representation from disk
        # This was created by extract_features_for_b3.py:
        #   1. Crop 12 players from middle frame
        #   2. Run through headless B3 model → (12, 2048)
        #   3. Max pool across players → (2048,)
        #   4. Save as .pt file
        feature_tensor = torch.load(item["file_path"])

        label = item["label"]

        return feature_tensor, label


# ============================================================================
# USAGE EXAMPLE (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example paths — adjust to your setup
    train_dataset = FeatureDataset(
        features_dir="/project/data/features/train",
        pkl_file="/project/data/annotations/annot_all.pkl",
    )

    # Check dataset size
    print(f"Train samples: {len(train_dataset)}")

    # Load first sample
    feat, label = train_dataset[0]
    print(f"Feature shape: {feat.shape}")  # torch.Size([2048])
    print(f"Feature dtype: {feat.dtype}")  # torch.float32
    print(f"Feature mean: {feat.mean():.4f}")  # ~0 (normalized)
    print(f"Feature std: {feat.std():.4f}")  # ~1 (normalized)
    print(f"Label: {label}")  # e.g., 1
    print(f"Label name: {list(train_dataset.category_to_idx.keys())[label]}")
    # → "r_pass"

    # Create DataLoader for training
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # One batch
    batch_feats, batch_labels = next(iter(train_loader))
    print(f"Batch features: {batch_feats.shape}")  # (32, 2048)
    print(f"Batch labels: {batch_labels.shape}")  # (32,)
