# Annotation
import os, pickle
from src.enums.PathEnums import Paths

class Annotation:
    def __init__(self):
        self.pkl_path = Paths.ANNOT_PKL.value # annot_path

    def load_pkl(self):
        with open(self.pkl_path, "rb") as f:
            videos_annot = pickle.load(f)
        return videos_annot

    def build_annotations(self):

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



if __name__ == "__main__":
    print("annotation...")
    
    loader = Annotation()
    data_annot, labels, categories_dct = loader.build_annotations()

    print(f"{len(labels)} labels: {labels}\n\n, {categories_dct}")


    for frame_dct in data_annot:
        for path, cat in frame_dct.items():
            print(frame_dct["path"], "==>", frame_dct["category"])
            break
        break
# python -m src.BaseLines.BaseLine1_ImageClassification.annotation