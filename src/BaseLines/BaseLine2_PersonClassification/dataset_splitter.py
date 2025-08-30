# dataset_splitter.py
import random, shutil, os, pickle
from pathlib import Path
from src.enums.PathEnums import Paths

class DatasetSplitter:

    def __init__(self):
        print("DatasetSplitter initialize...")

    def get_all_annotations(self):

        splits ={"train":[], "valid":[], "test":[], "all":[]}
        labels = set()
        # ----------------------------------------------------------------------------------------------------
        train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                    "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]

        val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]
        # ----------------------------------------------------------------------------------------------------

        with open(Paths.ANNOT_PKL.value, "rb") as f:
            videos_annot_dct = pickle.load(f)


        for video_id, clips in videos_annot_dct.items(): # each video
            video_id_str = str(video_id)

            if video_id_str in train_ids:
                split_name = "train"
            elif video_id_str in val_ids:
                split_name = "valid"
            else:
                split_name = "test"
            # print(video_id_str)

            for clip_id, clip_data in clips.items():     # each clip
                for frame_id, boxes in clip_data["frame_boxes_dct"].items():    # Frames
                    # print(len(boxes), video_id, clip_id, frame_id, boxes)  # 12 Box (Players)
                    frame_path = f"{Paths.VIDEOS_ROOT.value}/{video_id}/{clip_id}/{frame_id}.jpg"
                    # print(frame_path)
                    for box in boxes:
                        labels.add(box.category)

                        record = {
                            "frame path": frame_path,
                            "player box": box.box,
                            "category": box.category,
                        }

                        splits[split_name].append(record)
                        splits["all"].append(record)

        random.shuffle(splits["train"])
        random.shuffle(splits["valid"])
        random.shuffle(splits["test"])
        random.shuffle(splits["all"])

        return splits["all"], splits["train"], splits["valid"], splits["test"], sorted(list(labels))
# ===================================
def split_data():
    print("Start DatasetSplitter...\n")

    splitter = DatasetSplitter()
    all_data, train_split, valid_split, test_split, labels = splitter.get_all_annotations()
    
    print("labels: ", labels, "\n")
    print(f"len data: {len(all_data)} ||"
           f"train: {len(train_split)} ||" 
           f"valid: {len(valid_split)} ||" 
           f"test: {len(test_split)}")
    # Train: 231327, Valid: 143829, Test: 143406, All: 518562
    print("==="*50, "\n")

    return train_split, valid_split, test_split, labels
# ==============================================

if __name__ == "__main__":

    train_split, valid_split, test_split, labels = split_data()


# python -m src.BaseLines.BaseLine2_PersonClassification.dataset_splitter
