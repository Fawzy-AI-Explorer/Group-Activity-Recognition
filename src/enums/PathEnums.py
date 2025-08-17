from enum import Enum
from pathlib import Path


class Paths(Enum):

    DATASET_ROOT = "data/volleyball"
    VIDEOS_ROOT = f"{DATASET_ROOT}/volleyball_/videos"
    ANNOT_ROOT = f"{DATASET_ROOT}/volleyball_tracking_annotation/volleyball_tracking_annotation"

    ANNOT_PKL = f"datapkl/annot_all.pkl"

    # BaseLine 1 Paths

    OUTPUT_ROOT = Path("FramesDataB1")
    TRAIN_PATH = OUTPUT_ROOT / "train"
    VALID_PATH = OUTPUT_ROOT / "train"
    TEST_PATH = OUTPUT_ROOT / "train"





if __name__ == "__main__":
    print(Paths.DATASET_ROOT.value, Paths.DATASET_ROOT)
    print(Paths.VIDEOS_ROOT.value)
    print(Paths.ANNOT_ROOT.value)
    print(Paths.ANNOT_PKL.value)


    print(Paths.TRAIN_PATH.value, Paths.VALID_PATH.value, Paths.TEST_PATH.value)
# python -m src.enums.PathEnums
