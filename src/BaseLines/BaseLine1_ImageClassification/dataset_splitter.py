# dataset_splitter.py
import random, shutil, os
from pathlib import Path
from src.enums.PathEnums import Paths
from src.BaseLines.BaseLine1_ImageClassification.annotation import Annotation

class DatasetSplitter:
    def __init__(self):
        self.output_root = Paths.OUTPUT_ROOT.value

    def split(self, data_annot, train_ratio=0.6, valid_ratio=0.3):
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

    def save_split(self, data_split, split_path):
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


    annot_loader = Annotation()
    data_annot, labels, categories_dct = annot_loader.build_annotations()
    # print(f"{len(labels)} labels: {labels}\n\n, {categories_dct}")
    
 
    splitter = DatasetSplitter()

    train_data, valid_data, test_data = splitter.split(data_annot)
    # print(int(len(data_annot)*0.6) +  int(len(data_annot)*0.3) + int(len(data_annot)*0.1)) # 43470

    print("Train...") # 25973
    splitter.save_split(train_data, Paths.TRAIN_PATH.value)
    print("Valid...") # 12861
    splitter.save_split(valid_data, Paths.VALID_PATH.value)
    print("Test...") # 4300
    splitter.save_split(test_data, Paths.TEST_PATH.value)


# python -m src.BaseLines.BaseLine1_ImageClassification.dataset_splitter