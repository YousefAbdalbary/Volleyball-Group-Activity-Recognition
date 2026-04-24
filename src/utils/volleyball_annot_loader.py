import cv2
import os
import pickle
from typing import List
from boxinfo import BoxInfo

# ---------------------------------------------------------
# SMART PATHS: Detects if running on Kaggle or Ubuntu
# ---------------------------------------------------------
if os.path.exists("/kaggle/working"):
    print("Kaggle Environment Detected!")

    # NOTE: Change this to match your exact Kaggle dataset URL slug
    DATA_ROOT = "/kaggle/input/group-activity-recognition-volleyball"

    # Exact folder names based on standard Kaggle upload structures
    VIDEOS_ROOT = os.path.join(
        DATA_ROOT, "videos_sample"
    )  # Change to "videos" for full dataset
    ANNOT_ROOT = os.path.join(DATA_ROOT, "volleyball_tracking_annotation")

    # MUST save outputs to /kaggle/working/ because input is read-only
    PKL_FILE_PATH = "/kaggle/working/annot_all.pkl"

else:
    print("Local Ubuntu Environment Detected!")
    DATA_ROOT = "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data"
    VIDEOS_ROOT = os.path.join(DATA_ROOT, "sample")
    ANNOT_ROOT = os.path.join(DATA_ROOT, "annotations")
    PKL_FILE_PATH = os.path.join(ANNOT_ROOT, "annot_all.pkl")
# ---------------------------------------------------------


def load_tracking_annot(path):
    with open(path, "r") as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        # Create view from frame to boxes
        for player_ID, boxes_info in player_boxes.items():
            # Keep the middle 9 frames only (temporal windowing)
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct


def vis_clip(annot_path, video_dir):
    frame_boxes_dct = load_tracking_annot(annot_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(f"Processing visualization frames for {video_dir}...")

    for frame_id, boxes_info in frame_boxes_dct.items():
        img_path = os.path.join(video_dir, f"{frame_id}.jpg")
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found.")
            continue

        image = cv2.imread(img_path)

        for box_info in boxes_info:
            x1, y1, x2, y2 = box_info.box

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2
            )

        # --- DISABLED FOR KAGGLE ---
        # Kaggle will crash if you try to open a GUI window
        if not os.path.exists("/kaggle/working"):
            cv2.imshow("Volleyball Clip Viewer (Press Q to quit)", image)
            if cv2.waitKey(180) & 0xFF == ord("q"):
                break
        # ---------------------------

    if not os.path.exists("/kaggle/working"):
        cv2.destroyAllWindows()


def load_video_annot(video_annot):
    with open(video_annot, "r") as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(" ")[:2]
            clip_dir = items[0].replace(".jpg", "")
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct


def load_volleyball_dataset(videos_root, annot_root):
    videos_dirs = [
        d
        for d in os.listdir(videos_root)
        if os.path.isdir(os.path.join(videos_root, d))
    ]
    videos_dirs.sort()

    videos_annot = {}

    for idx, video_dir in enumerate(videos_dirs):
        video_dir_path = os.path.join(videos_root, video_dir)
        print(f"{idx+1}/{len(videos_dirs)} - Processing Dir {video_dir_path}")

        video_annot = os.path.join(video_dir_path, "annotations.txt")

        # Fallback if the video annotation text file is actually stored in the annot_root
        if not os.path.exists(video_annot):
            video_annot = os.path.join(annot_root, video_dir, "annotations.txt")

        # Skip if we still can't find the scene labels
        if not os.path.exists(video_annot):
            print(f"  -> Skipping: No annotations.txt found for video {video_dir}")
            continue

        clip_category_dct = load_video_annot(video_annot)

        clips_dir = [
            d
            for d in os.listdir(video_dir_path)
            if os.path.isdir(os.path.join(video_dir_path, d))
        ]
        clips_dir.sort()

        clip_annot = {}

        for clip_dir in clips_dir:
            clip_dir_path = os.path.join(video_dir_path, clip_dir)

            # Only process clips that have a corresponding scene label
            if clip_dir not in clip_category_dct:
                continue

            annot_file = os.path.join(
                annot_root, video_dir, clip_dir, f"{clip_dir}.txt"
            )
            if not os.path.exists(annot_file):
                continue

            frame_boxes_dct = load_tracking_annot(annot_file)

            clip_annot[clip_dir] = {
                "category": clip_category_dct[clip_dir],
                "frame_boxes_dct": frame_boxes_dct,
            }

        videos_annot[video_dir] = clip_annot

    return videos_annot


def create_pkl_version():
    print("Generating dataset dictionary...")
    videos_annot = load_volleyball_dataset(VIDEOS_ROOT, ANNOT_ROOT)

    print(f"Saving pickle file to {PKL_FILE_PATH}...")
    with open(PKL_FILE_PATH, "wb") as file:
        pickle.dump(videos_annot, file)
    print("Done!")


def test_pkl_version():
    if not os.path.exists(PKL_FILE_PATH):
        print("Pickle file not found. Run create_pkl_version() first.")
        return

    with open(PKL_FILE_PATH, "rb") as file:
        videos_annot = pickle.load(file)

    # Dynamically grab the first video, first clip, and first frame to test
    try:
        first_video = list(videos_annot.keys())[0]
        first_clip = list(videos_annot[first_video].keys())[0]
        first_frame = list(
            videos_annot[first_video][first_clip]["frame_boxes_dct"].keys()
        )[0]

        boxes: List[BoxInfo] = videos_annot[first_video][first_clip]["frame_boxes_dct"][
            first_frame
        ]

        print(f"Successfully loaded PKL!")
        print(
            f"Sample Data - Video: {first_video}, Clip: {first_clip}, Frame: {first_frame}"
        )
        print(f"Player Action: {boxes[0].category}")
        print(f"Bounding Box: {boxes[0].box}")
        print(
            f"Overall Scene Category: {videos_annot[first_video][first_clip]['category']}"
        )
    except IndexError:
        print(
            "Pickle file loaded, but it appears to be empty based on your sample data."
        )


if __name__ == "__main__":
    # 1. Create the PKL cache
    # If the file already exists on Kaggle from your upload, you can comment this out.
    if not os.path.exists(PKL_FILE_PATH):
        create_pkl_version()

    # 2. Test if the PKL loaded correctly
    test_pkl_version()

    # 3. Dynamically find the first clip to test logic (without GUI popups on Kaggle)
    try:
        first_vid = [
            d
            for d in os.listdir(VIDEOS_ROOT)
            if os.path.isdir(os.path.join(VIDEOS_ROOT, d))
        ][0]
        vid_path = os.path.join(VIDEOS_ROOT, first_vid)
        first_clip = [
            d for d in os.listdir(vid_path) if os.path.isdir(os.path.join(vid_path, d))
        ][0]

        annot_file = os.path.join(
            ANNOT_ROOT, first_vid, first_clip, f"{first_clip}.txt"
        )
        clip_dir_path = os.path.join(vid_path, first_clip)

        vis_clip(annot_file, clip_dir_path)
        print("Visualization logic passed safely!")
    except IndexError:
        print(
            f"Could not automatically find a video/clip inside {VIDEOS_ROOT} to test."
        )
