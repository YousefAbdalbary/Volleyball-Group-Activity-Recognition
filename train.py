import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Import the files you just built!
from src.dataset import VolleyballDataset
from src.models.baseline_b1 import BaselineB1ResNet

# --- 1. CONFIGURATION & PATHS ---

if os.path.exists("/kaggle/working"):
    print("Running on Kaggle...")
    # Exact name of your dataset slug
    DATA_ROOT = "/kaggle/input/group-activity-recognition-volleyball"
    OUTPUT_ROOT = "/kaggle/working"

    # Based on your Kaggle screenshot, the PKL is in the root of the dataset
    PKL_FILE = os.path.join(DATA_ROOT, "annot_all.pkl")

    # Change this to "videos" when you want to train on the full 50GB dataset!
    VIDEOS_DIR_NAME = "videos_sample"
else:
    print("Running on Local Ubuntu...")
    DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
    OUTPUT_ROOT = DATA_ROOT
    PKL_FILE = os.path.join(DATA_ROOT, "annotations", "annot_all.pkl")
    VIDEOS_DIR_NAME = "sample"

BATCH_SIZE = 16
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
    # --- 2. DEVICE SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # --- 3. DATA PREPARATION ---
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Find what videos are actually available in the target directory
    available_videos_path = os.path.join(DATA_ROOT, VIDEOS_DIR_NAME)
    available_videos = [
        d
        for d in os.listdir(available_videos_path)
        if os.path.isdir(os.path.join(available_videos_path, d))
    ]

    # Filter the available videos into train and val lists
    train_video_ids = [vid for vid in available_videos if vid in TRAIN_IDS]
    val_video_ids = [vid for vid in available_videos if vid in VAL_IDS]

    print(
        f"Found {len(train_video_ids)} training videos and {len(val_video_ids)} validation videos."
    )

    # Create Datasets
    train_dataset = VolleyballDataset(
        DATA_ROOT, PKL_FILE, train_video_ids, VIDEOS_DIR_NAME, transform
    )
    val_dataset = VolleyballDataset(
        DATA_ROOT, PKL_FILE, val_video_ids, VIDEOS_DIR_NAME, transform
    )

    # Use os.cpu_count() to dynamically set workers so Kaggle doesn't crash
    workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers
    )

    # We don't need to shuffle validation data
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=workers
    )

    # --- 4. MODEL, LOSS, AND OPTIMIZER ---
    model = BaselineB1ResNet(num_classes=8, fine_tune_all=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    # --- 5. THE TRAINING & VALIDATION LOOP ---
    for epoch in range(NUM_EPOCHS):

        # === TRAINING PHASE ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
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

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # === VALIDATION PHASE ===
        model.eval()  # Lock the weights
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Disable gradients for speed and memory saving
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # Handle case where val_loader might be empty if testing on a tiny sample
        if val_total > 0:
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
        else:
            val_acc, avg_val_loss = 0.0, 0.0

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        # === SAVE BEST MODEL ===
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_path = os.path.join(OUTPUT_ROOT, "baseline_b1_best.pth")
            torch.save(model.state_dict(), save_path)
            print(
                f"  -> Best model saved to {save_path} with Acc: {best_val_accuracy:.2f}%"
            )

    print("\nTraining Complete! Best Validation Accuracy:", best_val_accuracy)


if __name__ == "__main__":
    main()
