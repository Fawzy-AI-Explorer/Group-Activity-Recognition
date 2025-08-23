# dataset_splitter.py
import random, shutil, os, pickle
from pathlib import Path
from src.enums.PathEnums import Paths

class DatasetSplitter:
    def __init__(self, train_ratio=0.6, valid_ratio=0.1):
        # self.annotation_root = (Paths.ANNOT_PKL.value)
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio

    def get_all_annotations(self):

        with open(Paths.ANNOT_PKL.value, "rb") as f:
            videos_annot_dct = pickle.load(f)
        labels = set()
        all_annotations = []
        train_split = []
        valid_split = []
        i_tr = 0
        i_v = 0

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
                    all_annotations.append(
                        {
                            "path": frame_path, 
                            "category": category
                        }
                    )
                    if str(video_id) in train_ids:
                        i_tr+=1
                        train_split.append(
                            {
                                "path": frame_path, 
                                "category": category
                            }
                        )
                    if str(video_id) in val_ids:
                        i_v+=1
                        valid_split.append(
                            {
                                "path": frame_path, 
                                "category": category
                            }

                        )

                    
        print(i_tr, i_v, i_tr+i_v)
        return all_annotations, train_split, valid_split, valid_split, labels

    def split_dataset(self):
        all_annotations, train_split, valid_split, test_split, labels = self.get_all_annotations()
        # all_data_cpy = all_data.copy()
        # # random.shuffle(all_data)

        # n_total = len(all_data)
        # n_train = int(n_total * self.train_ratio)
        # n_valid = int(n_total * self.valid_ratio)

        # train_split = all_data[:n_train]
        # valid_split = all_data[n_train:n_train+n_valid]
        # test_split = all_data[n_train+n_valid:]

        return all_annotations, train_split, valid_split, test_split, sorted(list(labels))

# splitter = DatasetSplitter()
# train_split, valid_split, test_split, labels = splitter.split_dataset()

# len(data_annot), len(train_split), len(valid_split), len(test_split), len(labels), labels  # (43470, 26082, 4347, 13041, 8)

if __name__ == "__main__":

    splitter = DatasetSplitter(train_ratio=0.7, valid_ratio=0.1)
    all_data, train_split, valid_split, test_split, labels = splitter.split_dataset()

    print(f"len data: {len(all_data)} || train: {len(train_split)} || valid: {len(valid_split)} || test: {len(test_split)}")

    print(train_split[0], "\n")

    for frame_dct in all_data:
        for path, cat in frame_dct.items():
            print(frame_dct["path"], "==>", frame_dct["category"])
            break
        break
    
    


    # print(int(len(data_annot)*0.6) +  int(len(data_annot)*0.3) + int(len(data_annot)*0.1)) # 43470

    # print(Paths.TRAIN_PATH.value, Paths.VALID_PATH.value, Paths.TEST_PATH.value)

    # print(Paths.TRAIN_PATH.value, Paths.VALID_PATH.value, Paths.TEST_PATH.value)


# python -m src.BaseLines.BaseLine1_ImageClassification.dataset_splitter
