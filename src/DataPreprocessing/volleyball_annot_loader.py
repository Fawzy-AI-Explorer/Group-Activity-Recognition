import cv2
import os
import pickle
from typing import List, Dict
from tqdm import tqdm
from src.enums.PathEnums import Paths
from src.DataPreprocessing.box_info import BoxInfo
import matplotlib.pyplot as plt


class AnnotationLoader:
    """
    Class to load volleyball tracking annotations.
    """

    def __init__(self, debug: bool = False) -> None:
        """
        Initialize the AnnotationLoader with dataset paths.
        Attributes:
            videos_root (str): Path to the root directory of videos.
            annot_root (str): Path to the root directory of annotations.
        """

        self.videos_root = Paths.VIDEOS_ROOT.value
        self.annot_root = Paths.ANNOT_ROOT.value
        self.debug = debug

    def load_tracking_annot(self, path: str) -> dict:
        """
        Load tracking annotation from a clip file.
        Args:
            path (str): Path to the annotation clip file .
        Returns:
            dict: A dictionary where keys are frame IDs (9) and values are lists of (12) BoxInfo objects (Players).
        """

        # if not path.exists():
        #     raise FileNotFoundError(f"Annotation file not found: {path}") #todo : create this in function (helpers)

        with open(path, "r") as file:
            file_content = file.readlines()

            player_boxes_dct = {idx: [] for idx in range(12)}
            frame_boxes_dct = {}

            for idx, line in enumerate(
                tqdm(file_content, desc="Loading Annotations", unit="line")
            ):

                #  print(line)  # ! 12 Players * 20 Frames = 240 Lines =>> [0 0 0 0 0 1 1 1 1 2 2 2 ........ 11 11 11] each Player 20 times
                box_info = BoxInfo(line)

                if box_info.player_id > 11:
                    continue

                # print(box_info.player_ID) # ! [0 0 0 0 0 1 1 1 1 2 2 2 ........ 11 11 11] each Player 20 times
                player_boxes_dct[box_info.player_id].append(
                    box_info
                )  # ! 12 Players each one has 20 Boxes (Frames)

            # print(len(player_boxes_dct), len(player_boxes_dct[0]), len(player_boxes_dct[1]), len(player_boxes_dct[2]), len(player_boxes_dct[3]) ) # ! 12 20 20 20 20

            for (
                player_id,
                boxes_info,
            ) in tqdm(
                player_boxes_dct.items(), desc="Processing Players", unit="player"
            ):  # 12 (Players) * 20 (Frames) will be 9 frames later

                # keep the middle 9 frames only (enough for this task empirically)
                boxes_info = sorted(boxes_info, key=lambda box_info: box_info.frame_id)
                boxes_info = boxes_info[5:-6]

                player_boxes_dct[player_id] = boxes_info

                for box_info in boxes_info:  # 9 (Frames)
                    if (
                        box_info.frame_id not in frame_boxes_dct
                    ):  # ! we have 9 frames and each frame has 12 boxes (Players) each Frame appear 12 times
                        frame_boxes_dct[box_info.frame_id] = []

                    frame_boxes_dct[box_info.frame_id].append(box_info)  # 12 (Players)



            # print(len(frame_boxes_dct[38025]))  # ! 9 Frames each one contain 12 Info (Players)
            print(
                len(frame_boxes_dct), len(player_boxes_dct), len(player_boxes_dct[0])
            )  # 9  12  20
            return (
                frame_boxes_dct,  # 9 Frames each one contain 12 Info (box_info = Players)
                player_boxes_dct,  # 12 Players each one has 20 Boxes (Frames)
            )

    def vis_clip(self, annot_path, video_dir):
        """
        Visualize the annotations of a clip.
        Args:
            annot_path (str): Path to the annotation file.
            video_dir (str): Path to the directory containing video frames.
        """
        frame_boxes_dct, _ = self.load_tracking_annot(annot_path)
        # Dict {frame_id: List[BoxInfo]} 9 Frames each one contain 12 Info (Players)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for frame_id, boxes_info in frame_boxes_dct.items():  # ! 9 Frames

            img_path = os.path.join(video_dir, f"{frame_id}.jpg")
            image = cv2.imread(img_path)

            for box_info in boxes_info:  # ! 12 Players
                x1, y1, x2, y2 = box_info.box

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image, box_info.category, (x1, y1 - 10), font, 0.5, (0, 255, 0), 2
                )

            cv2.imshow("Image", image)
            cv2.waitKey(180)
        cv2.destroyAllWindows()

    def load_videoclips_annot(self, video_annot_file: str) -> dict:
        """
        Load video annotations from a file.
        Args:
            video_annot (str): Path to the video annotation file.
        Returns:
            dict: A dictionary where keys are clip directories and values are their categories.
        """
        with open(video_annot_file, "r") as file:
            clip_category_dct = {}

            file_content = file.readlines()
            for line in tqdm(
                file_content, desc="Loading Video Annotations", unit="line"
            ):
                items = line.strip().split(" ")[:2]
                clip_dir = items[0].replace(".jpg", "")
                clip_category_dct[clip_dir] = items[1]

            return clip_category_dct

    def load_volleyball_dataset(self):
        """
        Load the entire volleyball dataset.
        Returns:
            dict: A dictionary where keys are video directories and values are dictionaries of clip annotations.
            clip_annot: Dict {clip_dir: {'category': category, 'frame_boxes_dct': frame_boxes_dct, 'player_boxes': player_boxes}}
            frame_boxes_dct: Dict {frame_id: List[BoxInfo]} 9 Frames
            player_boxes: Dict {player_ID: List[BoxInfo]} 12 Players each one has 20 Boxes (Frames)
        """
        videos_dirs = os.listdir(self.videos_root)
        videos_dirs.sort()

        videos_annot_dct = {}

        # Iterate on each video and for each video iterate on each clip
        for idx, video_dir in enumerate(videos_dirs):
            video_dir_path = os.path.join(self.videos_root, video_dir)

            if not os.path.isdir(video_dir_path):
                continue

            print(f"{idx}/{len(videos_dirs)} - Processing Dir {video_dir_path}")

            video_annot = os.path.join(video_dir_path, "annotations.txt")
            clip_category_dct = self.load_videoclips_annot(
                video_annot
            )  # ! Dict {clip_dir: category}

            clips_dir = os.listdir(video_dir_path)
            clips_dir.sort()

            clip_annot = (
                {}
            )  # Dict {clip_dir: {'category': category, 'frame_boxes_dct': frame_boxes_dct}}

            for clip_dir in clips_dir:
                clip_dir_path = os.path.join(video_dir_path, clip_dir)

                if not os.path.isdir(clip_dir_path):
                    continue

                # print(f'\t{clip_dir_path}')
                assert clip_dir in clip_category_dct

                annot_file = os.path.join(
                    self.annot_root, video_dir, clip_dir, f"{clip_dir}.txt"
                )
                frame_boxes_dct, player_boxes = self.load_tracking_annot(annot_file)
                # vis_clip(annot_file, clip_dir_path)

                clip_annot[clip_dir] = {
                    "category": clip_category_dct[clip_dir],
                    "frame_boxes_dct": frame_boxes_dct,
                    "player_boxes": player_boxes,
                }

            videos_annot_dct[video_dir] = clip_annot

        return videos_annot_dct  # ! Dict {video_dir: {clip_dir: {'category': category, 'frame_boxes_dct': frame_boxes_dct}}}

    def create_pkl(self, volleyball_dataset=None):
        # You can use this function to create and save pkl version of the dataset
        # videos_root = f'{dataset_root}/videos'
        # annot_root = f'{dataset_root}/volleyball_tracking_annotation'

        videos_annot = (
            self.load_volleyball_dataset(self.videos_root, self.annot_root)
            if not volleyball_dataset
            else volleyball_dataset
        )

        with open(Paths.ANNOT_PKL.value, "wb") as file:
            pickle.dump(videos_annot, file)

    def load_pkl(self):
        with open(Paths.ANNOT_PKL.value, "rb") as file:
            videos_annot = pickle.load(file)

        boxes: List[BoxInfo] = videos_annot["0"]["13456"]["frame_boxes_dct"][13454]
        print(boxes[0].category)
        print(boxes[0].box)

        return videos_annot


if __name__ == "__main__":
    # clib_dir = r"4/24745"
    # annot_file = f'{dataset_root}/volleyball_tracking_annotation/volleyball_tracking_annotation/{clib_dir}/{os.path.basename(clib_dir)}.txt'
    # clip_dir_path = f'{dataset_root}/volleyball_/videos/{clib_dir}'
    annot_file = (
        r"E:\DATA SCIENCE\projects\dl\Group Activity Recognition\Data\38025.txt"
    )

    clip_dir_path = r"E:\DATA SCIENCE\projects\dl\Group Activity Recognition\Data\videos_sample\7\38025"

    loader = AnnotationLoader()
    loader.vis_clip(annot_file, clip_dir_path)
    # loader.create_pkl()  # Create pkl file
    # videos_annot = loader.load_pkl()  # Load pkl file
    print("END...")

# python -m src.DataPreprocessing.volleyball_annot_loader
