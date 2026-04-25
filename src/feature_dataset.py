import os
import pickle
import torch
from torch.utils.data import Dataset


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

        # 1. Map the 8 group activities to integers (Exactly like Baseline B1)
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
        self.data_list = []

        for pt_file in self.pt_files:
            # The file name format is "vid_clipid.pt" (e.g., "7_93635.pt")
            """
            vid_str, clip_str = pt_file.replace(".pt", "").split("_")
            # vid_str = "7"
            # clip_str = "93635"
            the vid_str is for video id
            the clip_str is for clip id


            -->Before split:
            like this --> "7_93635.pt"

            -->step by step show ::
            pt_file = "7_93635.pt"
            pt_file.replace(".pt", "") --> "7_93635"
            "7_93635".split("_") --> ["7", "93635"]
            so after split:
            vid_str = "7"
            clip_str = "93635"

            why i do that ??
            because the annotations.pkl file is like this:
            {
                "7": {
                    "93635": {
                        "category": "r-pass",
                        "frame_boxes_dct": {
                            "1": [BoxInfo(...)],
                            "2": [BoxInfo(...)],
                            "3": [BoxInfo(...)],
                        },
                    }
                }
            }
            so i need to get the vid_str and clip_str to get the label of the clip

            what is pt_file ??
            pt_file is a file that contains the features of the players
            it is like this:
            tensor([0.1, 0.2, 0.3, ...])

            where it come from ?
            it come from the extract_features.py file
            which is the output of the baseline_b3_person_model.py file


            where label come from ??
            it come from the annotations.pkl file
            which is the input of the extract_features.py






            """
            vid_str, clip_str = pt_file.replace(".pt", "").split("_")
            # Find the group label in the PKL dictionary
            clip_dict = self.annotations.get(vid_str) or self.annotations.get(
                int(vid_str)
            )
            """
            what clip_dict do ??
            clip_dict is a dictionary that contains the label of the clip
            it is like this:
            {
                "93635": {
                    "category": "r-pass",
                    "frame_boxes_dct": {
                        "1": [BoxInfo(...)],
                        "2": [BoxInfo(...)],
                        "3": [BoxInfo(...)],
                    },
                }
            }
            
            why i use .get(vid_str) or .get(int(vid_str)) ??
            because the annotations.pkl file is like this:
            {
                "7": {
                    "93635": {
                        "category": "r-pass",
                        "frame_boxes_dct": {
                            "1": [BoxInfo(...)],
                            "2": [BoxInfo(...)],
                            "3": [BoxInfo(...)],
                        },
                    }
                }
            }
            so i need to get the vid_str and clip_str to get the label of the clip

            -->>step by step show use of .get(vid_str) or .get(int(vid_str)  ::
                --> get(vid_str) is used to get the value of the key in the dictionary
                --> if the key is not found, it will return None
                --> get(int(vid_str)) is used to get the value of the key in the dictionary
                --> if the key is not found, it will return None
                
                example of return::
                annotations.get("7") --> {
                    "93635": {
                        "category": "r-pass",
                        "frame_boxes_dct": {
                            "1": [BoxInfo(...)],
                            "2": [BoxInfo(...)],
                            "3": [BoxInfo(...)],
                        },
                    }
                }
                annotations.get(int("7")) --> {
                    "93635": {
                        "category": "r-pass",
                        "frame_boxes_dct": {
                            "1": [BoxInfo(...)],
                            "2": [BoxInfo(...)],
                            "3": [BoxInfo(...)],
                        },
                    }
                }
                annotations.get("8") --> None
                annotations.get(int("8")) --> None
            

  
            """
            if clip_dict and clip_str in clip_dict:
                """
                explain::
                clip_dict[clip_str] is like this:
                {
                    "category": "r-pass",
                    "frame_boxes_dct": {
                        "1": [BoxInfo(...)],
                        "2": [BoxInfo(...)],
                        "3": [BoxInfo(...)],
                    },
                }

                Raw label is "r-pass" for right pass
                clean label is "r_pass" for right pass
                label_idx is 1 is used to get the features of the clip in the features_dir,
                and mapping the clean label to an integer
                _clean label is for removing the dash "-" from the raw label
                and category_to_idx is for mapping the clean label to an integer

                """
                raw_label = clip_dict[clip_str]["category"]
                clean_label = self._clean_label(raw_label)
                label_idx = self.category_to_idx[clean_label]

                """
                explain::
                self.data_list is a list of dictionaries, each dictionary contains the file path of the features and the label of the features
                the file path is like this:
                "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data/features/7_93635.pt"
                the label is like this:
                1

                how many filr path i have ?
                i have 12 player in each clip
                so i have 12 file path for each clip
                so i have 12 * number of clips file path
                
                """
                self.data_list.append(
                    {
                        "file_path": os.path.join(features_dir, pt_file),
                        "label": label_idx,
                    }
                )
            """ 
            this give me a list of dictionaries,
             each dictionary contains the file path of the features and the label of the features
            it is like this:
            [
                {
                    "file_path": "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data/features/7_93635.pt",
                    "label": 1,
                },
                {
                    "file_path": "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data/features/7_93636.pt",
                    "label": 1,
                },
                {
                    "file_path": "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data/features/7_93637.pt",
                    "label": 1,
                },
            ]
            for single clip there is 12 file path
            
            """
        print(f"Loaded {len(self.data_list)} extracted features from {features_dir}")

    def _clean_label(self, raw_label):
        # Cleans up messy text from the dataset (e.g., "Left Spike" -> "l-spike")

        """
        explain::
        - raw_label is like this: "Left Spike"
        - clean_label is like this: "l-spike"
        - label_idx is like this: 2
        - Before clean is like this --> "Left Spike"
        - After clean is like this --> "l-spike"

        Q- why i do that ??
        - because the model is trained on this format
        Q- how did i know that ??
        - i checked the annotations.pkl file
        Q- why format(left spike) ??
        - because the dataset is created in this format
        Q- so what i do ?
        i replace the space with dash and the first letter of the action with lowercase

        """

        # Bulletproof: .lower() happens FIRST
        label = (
            raw_label.lower()
            .strip()
            .replace(" ", "_")
            .replace("left", "l")
            .replace("right", "r")
        )

        # The hyphen hacks
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
        feature_tensor = torch.load(item["file_path"])
        """ feature_tensor is like this:
        tensor([0.1, 0.2, 0.3, ...])
        shape is (2048,)
        use item["file_path"] to get the file path , item is for single clip, it 
        will be like this: "7_93635.pt" amd will open it using torch.load() , be like this 
        use torch.load() to load the features
        """

        label = item["label"]
        """ label is like this:
        1
        shape is (1,)
        """

        return feature_tensor, label

        """
        finally i have a dataset that contains the features of the players and the labels of the features
        it is like this:
        [
            (tensor([0.1, 0.2, 0.3, ...]), 1),
            (tensor([0.4, 0.5, 0.6, ...]), 2),
            (tensor([0.7, 0.8, 0.9, ...]), 3),
        ]

        step of what happen in __getitem__:
        1. get the item from the data_list
        2. load the features from the file path
        3. return the features and the label
        

        step of what happen in __init__:
        1. get the features_dir and annotations_path
        2. load the annotations from the annotations_path
        3. get the pt files from the features_dir
        4. for each pt file:
            a. get the vid_str and clip_str
            b. get the label of the clip
            c. add the file path and the label to the data_list
        
       data_list is like this:
        [
            {
                "file_path": "/media/yousef-abdalbary/NewVolume/download/Deep Learning/volleyball_project/data/features/7_93635.pt",
                "label": 1,
            }
        ]

        summary:
        i make this code to load the features of the players and the labels of the features
        
        is it like mapping between feature represeantion of 12 player 
        And the Action or main action of the clip or group.
            
        """
