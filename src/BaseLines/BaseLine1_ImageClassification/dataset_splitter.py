# dataset_splitter.py
import random, shutil, os, pickle
from pathlib import Path
from src.enums.PathEnums import Paths

class DatasetSplitter:

    def __init__(self):
        print("DatasetSplitter initialize...")

    def get_all_annotations(self):

        with open(Paths.ANNOT_PKL.value, "rb") as f:
            videos_annot_dct = pickle.load(f)

        labels = set()
        all_annotations = []
        train_split = []
        valid_split = []
        test_split = []
        i_tr, i_v, i_ts, i_all = 0, 0, 0, 0

        train_ids = ["1", "3", "6", "7", "10", "13", "15", "16", "18", "22", "23", "31",
                    "32", "36", "38", "39", "40", "41", "42", "48", "50", "52", "53", "54"]


        val_ids = ["0", "2", "8", "12", "17", "19", "24", "26", "27", "28", "30", "33", "46", "49", "51"]

        for video_id, clips in videos_annot_dct.items(): # each video

            for clip_id, clip_data in clips.items():     # each clip

                category = clip_data['category']
                labels.add(category)

                for frame_id, boxes in clip_data["frame_boxes_dct"].items():    # Frames

                    # if str(frame_id) == str(clip_id) :
                        # print(f"===: {video_id} || {clip_id} || {frame_id}, {category}")
                    frame_path = f"{Paths.VIDEOS_ROOT.value}/{video_id}/{clip_id}/{frame_id}.jpg"
                    i_all+=1
                    all_annotations.append({"path": frame_path,   "category": category })
                        
                    if str(video_id) in train_ids:
                        i_tr+=1
                        # if i_tr == 2:
                        #     print(f"{i_tr}_train: path: {frame_path}, category: {category}")
                            # plot_image(frame_path, category)
                        train_split.append({"path": frame_path,   "category": category })
                    elif str(video_id) in val_ids:
                        i_v+=1
                        # if i_v == 3:
                        #     print(f"{i_v}_valid: path: {frame_path}, category: {category}")
                            # plot_image(frame_path, category)
                        valid_split.append({"path": frame_path,   "category": category })
                    else:
                        i_ts+=1
                        # if i_ts == 1:
                        #     print(f"{i_ts}_test: path: {frame_path}, category: {category}")
                        #     plot_image(frame_path, category)
                        test_split.append({"path": frame_path,   "category": category })

        print(i_tr, i_v, i_ts, i_tr+i_v+i_ts, i_all)


        random.shuffle(train_split)
        random.shuffle(valid_split)
        random.shuffle(test_split)
        random.shuffle(all_annotations)

        return all_annotations, train_split, valid_split, test_split, sorted(list(labels))
# ===================================
def split_data():
    print("Start DatasetSplitter...\n")

    splitter = DatasetSplitter()
    all_data, train_split, valid_split, test_split, labels = splitter.get_all_annotations()
    
    print("labels: ", labels, "\n")
    print(f"len data: {len(all_data)} || train: {len(train_split)} || valid: {len(valid_split)} || test: {len(test_split)}")
    print("==="*50, "\n")

    return train_split, valid_split, test_split, labels
# ==============================================


if __name__ == "__main__":

    train_split, valid_split, test_split, labels = split_data()


# python -m src.BaseLines.BaseLine1_ImageClassification.dataset_splitter
