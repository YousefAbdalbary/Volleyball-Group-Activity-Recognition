import os
import sys
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# --- THE PICKLE FIX ---
# Tell Python to look inside the 'utils' folder right next to this file
# so pickle knows what the 'BoxInfo' class is when it loads the cache.
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
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
        """
        Args:
            data_root (str): The root path to your 'data/' folder.
            pkl_file (str): The path to your generated 'annot_all.pkl' file.
            split_video_ids (list): A list of video IDs (strings) to include in this dataset.
            video_dir_name (str): The name of the folder containing the JPGs (e.g., 'sample' or 'videos_sample').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_root = data_root
        self.video_dir_name = video_dir_name
        self.transform = transform

        print(f"Loading annotations from {pkl_file}...")
        with open(pkl_file, "rb") as f:
            self.annotations = pickle.load(f)

        # 1. Flatten the nested dictionary into a simple list so PyTorch can index it easily.
        self.clip_list = []
        for vid in split_video_ids:
            # Check if the video ID actually exists in our parsed annotations
            if vid in self.annotations:
                for clip_id, clip_data in self.annotations[vid].items():
                    self.clip_list.append(
                        {
                            "video_id": vid,
                            "clip_id": clip_id,
                            "category": clip_data["category"],
                            "frames": list(clip_data["frame_boxes_dct"].keys()),
                        }
                    )

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
        """Returns the total number of clips in this dataset."""
        return len(self.clip_list)

    def _clean_label(self, raw_label):
        """
        Helper function to normalize the raw string labels.
        Forces them to match our dictionary keys exactly.
        """
        label = (
            raw_label.strip()
            .replace(" ", "_")
            .replace("Left", "l")
            .replace("Right", "r")
            .lower()
        )

        # Handle specific dash/underscore inconsistencies in the dataset
        if label == "l_pass":
            return "l-pass"
        if label == "r_pass":
            return "r-pass"
        if label == "l_spike":
            return "l-spike"

        return label

    def __getitem__(self, idx):
        """
        Fetches the image and label for a given index.
        For Baseline B1, we only want the MIDDLE frame of the clip.
        """
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

        # Load the image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # Apply PyTorch transformations (Resize, Normalize, ToTensor)
        if self.transform:
            image = self.transform(image)

        # Get the integer label
        clean_label_str = self._clean_label(clip_info["category"])
        label_idx = self.category_to_idx[clean_label_str]

        # Return tuple of (Tensor, Integer)
        return image, label_idx


# ==========================================
# TEST BLOCK: Run this to verify your code!
# ==========================================
if __name__ == "__main__":
    # 1. Define your paths based on your environment
    if os.path.exists("/kaggle/working"):
        DATA_ROOT = (
            "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball"
        )
        OUTPUT_ROOT = "/kaggle/working"
        PKL_FILE = os.path.join(DATA_ROOT, "annot_all.pkl")
        VIDEOS_DIR_NAME = "videos_sample"
    else:
        DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
        OUTPUT_ROOT = DATA_ROOT
        PKL_FILE = os.path.join(DATA_ROOT, "annotations", "annot_all.pkl")
        VIDEOS_DIR_NAME = "sample"

    # 2. Define standard ImageNet transforms for ResNet50
    test_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3. Create a fake "split" using whatever video folders exist in your directory
    sample_videos_path = os.path.join(DATA_ROOT, VIDEOS_DIR_NAME)

    if not os.path.exists(sample_videos_path):
        print(f"Error: Could not find video path {sample_videos_path}")
        available_videos = []
    else:
        available_videos = [
            d
            for d in os.listdir(sample_videos_path)
            if os.path.isdir(os.path.join(sample_videos_path, d))
        ]

    if not available_videos:
        print(f"Error: No video folders found in {sample_videos_path}")
    elif not os.path.exists(PKL_FILE):
        print(
            f"Error: {PKL_FILE} not found. Did you run the create_pkl_version() function in your loader script?"
        )
    else:
        print(f"Testing with available videos: {available_videos}")

        # 4. Instantiate the Dataset
        test_dataset = VolleyballDataset(
            data_root=DATA_ROOT,
            pkl_file=PKL_FILE,
            split_video_ids=available_videos,
            video_dir_name=VIDEOS_DIR_NAME,
            transform=test_transform,
        )

        # 5. Fetch the very first item
        try:
            image_tensor, label = test_dataset[0]

            print("\n--- TEST SUCCESSFUL ---")
            print(
                f"Image Tensor Shape: {image_tensor.shape}"
            )  # Should be torch.Size([3, 224, 224])
            print(
                f"Image Tensor DataType: {image_tensor.dtype}"
            )  # Should be torch.float32
            print(f"Label Integer: {label}")  # Should be a number 0-7

            # Reverse map the label to prove it works
            idx_to_category = {v: k for k, v in test_dataset.category_to_idx.items()}
            print(f"Label String: {idx_to_category[label]}")

        except Exception as e:
            print(f"\n--- TEST FAILED ---")
            print(f"Error: {e}")
