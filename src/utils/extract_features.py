import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Ensure volleyball_annot_loader.py is in the same folder
from volleyball_annot_loader import load_tracking_annot

# ---------------------------------------------------------
# SMART PATHS: Detects if running on Kaggle or Ubuntu
# ---------------------------------------------------------
if os.path.exists("/kaggle/working"):
    print("Kaggle Environment Detected!")
    DATA_ROOT = "/kaggle/input/group-activity-recognition-volleyball"
    VIDEOS_ROOT = os.path.join(
        DATA_ROOT, "videos_sample"
    )  # Change to "videos" for full dataset
    ANNOT_ROOT = os.path.join(DATA_ROOT, "volleyball_tracking_annotation")

    # Kaggle MUST save generated feature arrays to the working directory
    OUTPUT_ROOT_BASE = "/kaggle/working/features"
else:
    print("Local Ubuntu Environment Detected!")
    DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
    VIDEOS_ROOT = os.path.join(DATA_ROOT, "sample")
    ANNOT_ROOT = os.path.join(DATA_ROOT, "annotations")
    OUTPUT_ROOT_BASE = os.path.join(DATA_ROOT, "features")
# ---------------------------------------------------------


def check():
    print("torch: version", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA is available.")
        num_devices = torch.cuda.device_count()
        print(f"Number of GPU devices: {num_devices}")
        for i in range(num_devices):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else "CPU"
    print(f"Current device: {current_device}")


def prepare_model(image_level=False):
    if image_level:
        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ResNet50
    model = models.resnet50(pretrained=True)
    # Remove classification head to output 2048-dim features
    model = nn.Sequential(*(list(model.children())[:-1]))

    model.to(device)
    model.eval()

    return model, preprocess, device


def extract_features(
    clip_dir_path, annot_file, output_file, model, preprocess, device, image_level=False
):
    """
    what is he goal of extract_features function ??
    --> it is to extract the features of the players in the clip

    what is this mean ??
    it mean that we will take the middle frame of the clip
    and extract the features of the players in the clip
    and save them to a file

    """
    frame_boxes = load_tracking_annot(annot_file)

    with torch.no_grad():
        for frame_id, boxes_info in frame_boxes.items():
            try:
                img_path = os.path.join(clip_dir_path, f"{frame_id}.jpg")
                if not os.path.exists(img_path):
                    continue

                image = Image.open(img_path).convert("RGB")

                if image_level:
                    preprocessed_image = preprocess(image).unsqueeze(0).to(device)
                    dnn_repr = model(preprocessed_image)
                    dnn_repr = dnn_repr.view(1, -1)
                else:
                    preprocessed_images = []
                    for box_info in boxes_info:
                        x1, y1, x2, y2 = box_info.box

                        # Safety check for boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        cropped_image = image.crop((x1, y1, x2, y2))

                        # Completely disabled cv2.imshow for Kaggle safety
                        preprocessed_images.append(
                            preprocess(cropped_image).unsqueeze(0)
                        )

                    if not preprocessed_images:
                        continue

                    preprocessed_images = torch.cat(preprocessed_images).to(device)
                    dnn_repr = model(preprocessed_images)
                    dnn_repr = dnn_repr.view(len(preprocessed_images), -1)

                # Move tensor to CPU before converting to numpy
                feature_array = dnn_repr.cpu().numpy()

                # Save per frame
                frame_output_file = output_file.replace(".npy", f"_{frame_id}.npy")
                np.save(frame_output_file, feature_array)

            except Exception as e:
                print(f"An error occurred in frame {frame_id}: {e}")


if __name__ == "__main__":
    check()

    # NOTE: Set to True for Baseline B4. Set to False for Baseline B5/B7.
    image_level = False
    model, preprocess, device = prepare_model(image_level)

    feature_type = "image-level" if image_level else "box-level"
    output_root = os.path.join(OUTPUT_ROOT_BASE, feature_type, "resnet50")

    videos_dirs = [
        d
        for d in os.listdir(VIDEOS_ROOT)
        if os.path.isdir(os.path.join(VIDEOS_ROOT, d))
    ]
    videos_dirs.sort()

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(VIDEOS_ROOT, video_dir)
        print(f"{idx+1}/{len(videos_dirs)} - Processing Dir {video_dir_path}")

        clips_dir = [
            d
            for d in os.listdir(video_dir_path)
            if os.path.isdir(os.path.join(video_dir_path, d))
        ]
        clips_dir.sort()

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            annot_file = os.path.join(
                ANNOT_ROOT, video_dir, clip_dir, f"{clip_dir}.txt"
            )

            output_dir = os.path.join(output_root, video_dir, clip_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_file_base = os.path.join(output_dir, "features.npy")

            if os.path.exists(annot_file):
                extract_features(
                    clip_dir_path,
                    annot_file,
                    output_file_base,
                    model,
                    preprocess,
                    device,
                    image_level=image_level,
                )
