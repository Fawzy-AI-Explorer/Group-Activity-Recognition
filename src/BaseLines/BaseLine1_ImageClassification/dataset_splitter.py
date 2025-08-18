# # dataset_splitter.py
# import random, shutil, os
# from pathlib import Path
# from src.enums.PathEnums import Paths
# from src.BaseLines.BaseLine1_ImageClassification.annotation import Annotation

# class DatasetSplitter:
#     def __init__(self):
#         self.output_root = Paths.OUTPUT_ROOT.value

#     def split(self, data_annot, train_ratio=0.6, valid_ratio=0.3):
#         random.shuffle(data_annot)
#         n_total = len(data_annot)
#         n_train = int(train_ratio * n_total)
#         n_valid = int(valid_ratio * n_total)
#         # print(n_total, n_train, n_valid, n_total-n_train-n_valid)
#         # 26082 || 13041 || 4347
#         return (
#             data_annot[:n_train],
#             data_annot[n_train:n_train+n_valid],
#             data_annot[n_train+n_valid:]
#         )

#     def save_split(self, data_split, split_path):
#         split_path.mkdir(parents=True, exist_ok=True)
#         count = 0
#         count_override = 0
#         for item in data_split:

#             src = Path(item["path"])
#             label = item["category"]
#             dst_dir = split_path / label
#             dst_dir.mkdir(parents=True, exist_ok=True)
#             vid_num = Path(src).parents[1].name
#             dst = dst_dir / f"vid_{vid_num}_{os.path.basename(src)}"
            
#             try:
#                 if dst.exists():
#                     num = random.randint(1, 10)
#                     dst = dst_dir / f"vid_{vid_num}_{num}_{src.name}"
#                     count_override+=1

#                 shutil.copy(src, dst)
#                 count += 1

#             # try:
#             #     if not dst.exists():
#             #         shutil.copy(src, dst)
#             #         count += 1
#             #     else:
#             #         num = random.randint(1, 10)
#             #         dst = dst_dir / f"vid_{num}_{vid_num}_{os.path.basename(src)}"
#             #         shutil.copy(src, dst)
#             #         count += 1

#             except Exception as e:
#                 print(f"Error copying {src} → {dst}: {e}")
#         print(count, count_override, count_override+count)


# if __name__ == "__main__":


#     annot_loader = Annotation()
#     data_annot, labels, categories_dct = annot_loader.build_annotations()
#     # print(f"{len(labels)} labels: {labels}\n\n, {categories_dct}")
    
 
#     splitter = DatasetSplitter()

#     train_data, valid_data, test_data = splitter.split(data_annot)
#     # print(int(len(data_annot)*0.6) +  int(len(data_annot)*0.3) + int(len(data_annot)*0.1)) # 43470

#     print("Train...") # 25973
#     splitter.save_split(train_data, Paths.TRAIN_PATH.value)
#     print("Valid...") # 12861
#     splitter.save_split(valid_data, Paths.VALID_PATH.value)
#     print("Test...") # 4300
#     splitter.save_split(test_data, Paths.TEST_PATH.value)

# dataset_splitter.py
import random, shutil, os, pickle
from pathlib import Path
from src.enums.PathEnums import Paths
from src.BaseLines.BaseLine1_ImageClassification.annotation import Annotation

class DatasetSplitter:
    """
    Load data and split the dataset into train, validation, and test sets.
    """

    def __init__(self) -> None:
        """
        Initialize the DatasetSplitter.
        Attributes:
            output_root (str): The root directory for output data.
            pkl_path (str): The path to the pickle file containing annotations.

        """
        self.output_root = Paths.OUTPUT_ROOT.value
        self.pkl_path = Paths.ANNOT_PKL.value # annot_path

    def load_pkl(self) -> dict:
        """
        Load the pickle file containing video annotations.
        returns:
            dict: A dictionary containing video annotations.
        """
        with open(self.pkl_path, "rb") as f:
            videos_annot = pickle.load(f)
        return videos_annot


    def build_annotations(self):
        """
        Build annotations from the loaded pickle file.
        returns:
            data_annot: A list of dictionaries containing frame paths and their categories.
            labels: A list of unique category labels.
            categories_dct: A dictionary mapping category labels to their indices.
        """

        videos_annot_dct = self.load_pkl()
        # print(len(videos_annot_dct)) # 55
        data_annot, labels = [], set()

        for video_id, clips in videos_annot_dct.items(): # each video

            for clip_id, clip_data in clips.items():     # each clip

                category = clip_data['category']
                labels.add(category)

                for frame_id, boxes in clip_data["frame_boxes_dct"].items():    # Frames
                    frame_path = f"{Paths.VIDEOS_ROOT.value}/{video_id}/{clip_id}/{frame_id}.jpg"
                    data_annot.append(
                        {
                            "path": frame_path, 
                            "category": category
                        }
                    )
        labels = sorted(labels)
        categories_dct = {label: idx for idx, label in enumerate(labels)}

        # categories_dct = {
        #     'l-pass': 0,
        #     'r-pass': 1,
        #     'l-spike': 2,
        #     'r_spike': 3,
        #     'l_set': 4,
        #     'r_set': 5,
        #     'l_winpoint': 6,
        #     'r_winpoint': 7
        # }
        # labels = list(categories_dct.keys())

        return data_annot, labels, categories_dct


    def split(self, data_annot, train_ratio=0.6, valid_ratio=0.3) -> tuple:
        """
        Split the dataset into train, validation, and test sets.
        Args:
            data_annot (list): The annotated data to split.
            train_ratio (float): The proportion of data to use for training.
            valid_ratio (float): The proportion of data to use for validation.
        Returns:
            tuple: A tuple containing the train, validation, and test splits.
        """
        random.shuffle(data_annot)
        n_total = len(data_annot)
        n_train = int(train_ratio * n_total)
        n_valid = int(valid_ratio * n_total)
        # print(n_total, n_train, n_valid, n_total-n_train-n_valid)
        # 26082 || 13041 || 4347
        return (
            data_annot[:n_train],
            data_annot[n_train:n_train+n_valid],
            data_annot[n_train+n_valid:]
        )

    def save_split(self, data_split, split_path) -> None:
        """
        Save the split dataset to the specified path.
        """
        split_path.mkdir(parents=True, exist_ok=True)
        count = 0
        count_override = 0
        for item in data_split:

            src = Path(item["path"])
            label = item["category"]
            dst_dir = split_path / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            vid_num = Path(src).parents[1].name
            dst = dst_dir / f"vid_{vid_num}_{os.path.basename(src)}"
            
            try:
                if dst.exists():
                    num = random.randint(1, 10)
                    dst = dst_dir / f"vid_{vid_num}_{num}_{src.name}"
                    count_override+=1

                shutil.copy(src, dst)
                count += 1

            # try:
            #     if not dst.exists():
            #         shutil.copy(src, dst)
            #         count += 1
            #     else:
            #         num = random.randint(1, 10)
            #         dst = dst_dir / f"vid_{num}_{vid_num}_{os.path.basename(src)}"
            #         shutil.copy(src, dst)
            #         count += 1

            except Exception as e:
                print(f"Error copying {src} → {dst}: {e}")
        print(count, count_override, count_override+count)


if __name__ == "__main__":


    annot_loader = DatasetSplitter()
    data_annot, labels, categories_dct = annot_loader.build_annotations()

    # print(f"{len(labels)} labels: {labels}\n\n, {categories_dct}")
    for frame_dct in data_annot:
        for path, cat in frame_dct.items():
            print(frame_dct["path"], "==>", frame_dct["category"])
            break
        break
    
    train_data, valid_data, test_data = annot_loader.split(data_annot)
    # print(int(len(data_annot)*0.6) +  int(len(data_annot)*0.3) + int(len(data_annot)*0.1)) # 43470

    print("Train...") # 25973
    annot_loader.save_split(train_data, Paths.TRAIN_PATH.value)
    print("Valid...") # 12861
    annot_loader.save_split(valid_data, Paths.VALID_PATH.value)
    print("Test...") # 4300
    annot_loader.save_split(test_data, Paths.TEST_PATH.value)


# python -m src.BaseLines.BaseLine1_ImageClassification.dataset_splitter








# # python -m src.BaseLines.BaseLine1_ImageClassification.dataset_splitter