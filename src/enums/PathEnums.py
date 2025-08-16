from enum import Enum

class Paths(Enum):

    DATASET_ROOT = "data/volleyball"
    VIDEOS_ROOT = f"{DATASET_ROOT}/volleyball_/videos"
    ANNOT_ROOT = f"{DATASET_ROOT}/volleyball_tracking_annotation/volleyball_tracking_annotation"

    ANNOT_PKL = f"datapkl/annot_all.pkl"


if __name__ == "__main__":
    print(Paths.DATASET_ROOT.value, Paths.DATASET_ROOT)
    print(Paths.VIDEOS_ROOT.value)
    print(Paths.ANNOT_ROOT.value)
    print(Paths.ANNOT_PKL.value)
# python -m src.enums.PathEnums
