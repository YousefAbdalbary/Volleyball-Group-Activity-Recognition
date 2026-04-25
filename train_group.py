import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom components
from src.feature_dataset import FeatureDataset
from src.models.baseline_b3 import BaselineB3GroupModel

# --- 1. CONFIGURATION ---
if os.path.exists("/kaggle/working"):
    print("Running on Kaggle...")
    # Point directly to the folder where extract_features.py saved the data
    FEATURES_ROOT = "/kaggle/working/features"
    PKL_FILE = "/kaggle/input/datasets/sherif31/group-activity-recognition-volleyball/annot_all.pkl"
    OUTPUT_ROOT = "/kaggle/working"
else:
    print("Running on Local Ubuntu...")
    DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
    FEATURES_ROOT = os.path.join(DATA_ROOT, "features")
    PKL_FILE = os.path.join(DATA_ROOT, "annotations", "annot_all.pkl")
    OUTPUT_ROOT = DATA_ROOT

BATCH_SIZE = (
    64  # We can process 64 clips at a time because they are just tiny 1D arrays!
)
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50  # We run more epochs because MLP training is incredibly fast.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Head Coach on device: {device}")

    # --- 2. DATA LOADERS ---
    train_dir = os.path.join(FEATURES_ROOT, "train")
    val_dir = os.path.join(FEATURES_ROOT, "val")

    print("Loading Training Features...")
    train_dataset = FeatureDataset(train_dir, PKL_FILE)

    print("Loading Validation Features...")
    val_dataset = FeatureDataset(val_dir, PKL_FILE)

    # Note: We don't need num_workers for tiny .pt files, 0 is actually faster here!
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. THE "HEAD COACH" MODEL ---
    # This is the Multi-Layer Perceptron from baseline_b3.py
    model = BaselineB3GroupModel(input_features=2048, num_classes=8)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0

    # --- 4. THE LIGHTNING-FAST TRAINING LOOP ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Notice we are passing 'features' instead of 'images'
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
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

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_path = os.path.join(OUTPUT_ROOT, "baseline_b3_group_model.pth")
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Final Model saved as baseline_b3_group_model.pth")


if __name__ == "__main__":
    main()
