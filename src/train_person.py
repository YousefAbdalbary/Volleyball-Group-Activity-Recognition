import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Import your new dataset and recycle your old model!
from src.crop_dataset import PlayerCropDataset
from src.models.baseline_b3 import BaselineB3PersonModel

# --- 1. CONFIGURATION & PATHS ---
if os.path.exists("/kaggle/working"):
    print("Running on Kaggle...")
    DATA_ROOT = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball"
    OUTPUT_ROOT = "/kaggle/working"
    PKL_FILE = os.path.join(DATA_ROOT, "annot_all.pkl")

    # Automatically handle the nested videos folder on Kaggle
    if os.path.exists(os.path.join(DATA_ROOT, "videos", "videos")):
        VIDEOS_DIR_NAME = "videos/videos"
    else:
        VIDEOS_DIR_NAME = "videos"
else:
    print("Running on Local Ubuntu...")
    DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
    OUTPUT_ROOT = DATA_ROOT
    PKL_FILE = os.path.join(DATA_ROOT, "annotations", "annot_all.pkl")
    VIDEOS_DIR_NAME = "sample"


# We can use a slightly larger batch size since these are small crops
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10


# Dr. Mostafa's Official Dataset Split
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    # --- 2. DATA PREPARATION ---
    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224)
            ),  # Stretch the tiny player rectangle into a standard square
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    available_videos_path = os.path.join(DATA_ROOT, VIDEOS_DIR_NAME)
    available_videos = [
        d
        for d in os.listdir(available_videos_path)
        if os.path.isdir(os.path.join(available_videos_path, d))
    ]

    train_video_ids = [vid for vid in available_videos if vid in TRAIN_IDS]
    val_video_ids = [vid for vid in available_videos if vid in VAL_IDS]

    # Create the New Datasets
    print("--- Initializing Training Dataset ---")
    train_dataset = PlayerCropDataset(
        DATA_ROOT, PKL_FILE, train_video_ids, VIDEOS_DIR_NAME, transform
    )
    print("--- Initializing Validation Dataset ---")
    val_dataset = PlayerCropDataset(
        DATA_ROOT, PKL_FILE, val_video_ids, VIDEOS_DIR_NAME, transform
    )

    workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers
    )

    # --- 3. THE "SCOUT" MODEL ---
    # CRITICAL CHANGE: We tell ResNet50 to output 9 classes instead of 8!
    model = BaselineB3PersonModel(num_classes=9, fine_tune_all=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # --- 3. MODEL SETUP ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    # --- 4. TRAINING LOOP ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        # Save this specific model so we can use it in Phase B
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_path = os.path.join(OUTPUT_ROOT, "baseline_b3_person_model.pth")
            torch.save(model.state_dict(), save_path)
            print(
                f"  -> Best Scout Model saved to {save_path} with Acc: {best_val_accuracy:.2f}%"
            )


if __name__ == "__main__":
    main()
