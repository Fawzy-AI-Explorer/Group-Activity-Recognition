# dataset_splitter.py
import random, shutil, os, pickle
from pathlib import Path
from src.enums.PathEnums import Paths
from src.BaseLines.BaseLine1_ImageClassification.annotation import Annotation

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
        for video_id, clips in videos_annot_dct.items(): # each video

            for clip_id, clip_data in clips.items():     # each clip

                category = clip_data['category']
                labels.add(category)

                for frame_id, boxes in clip_data["frame_boxes_dct"].items():    # Frames
                    frame_path = f"{Paths.VIDEOS_ROOT.value}/{video_id}/{clip_id}/{frame_id}.jpg"
                    all_annotations.append(
                        {
                            "path": frame_path, 
                            "category": category
                        }
                    )
        return all_annotations, labels

    def split_dataset(self):
        all_data, labels = self.get_all_annotations()
        all_data_cpy = all_data.copy()
        random.shuffle(all_data)

        n_total = len(all_data)
        n_train = int(n_total * self.train_ratio)
        n_valid = int(n_total * self.valid_ratio)

        train_split = all_data[:n_train]
        valid_split = all_data[n_train:n_train+n_valid]
        test_split = all_data[n_train+n_valid:]

        return all_data_cpy, train_split, valid_split, test_split, sorted(list(labels))

# splitter = DatasetSplitter()
# train_split, valid_split, test_split, labels = splitter.split_dataset()

# len(data_annot), len(train_split), len(valid_split), len(test_split), len(labels), labels  # (43470, 26082, 4347, 13041, 8)

if __name__ == "__main__":

    splitter = DatasetSplitter()
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

    # print("Train...") # 25973
    # annot_loader.save_split(train_data, Paths.TRAIN_PATH.value)
    # print("Valid...") # 12861
    # annot_loader.save_split(valid_data, Paths.VALID_PATH.value)
    # print("Test...") # 4300
    # annot_loader.save_split(test_data, Paths.TEST_PATH.value)
    # print(Paths.TRAIN_PATH.value, Paths.VALID_PATH.value, Paths.TEST_PATH.value)


# python -m src.BaseLines.BaseLine1_ImageClassification.dataset_splitter
