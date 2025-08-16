from src.DataPreprocessing.volleyball_annot_loader import AnnotationLoader
from src.enums.PathEnums import Paths


loader = AnnotationLoader()
videos_annot_dct = loader.load_pkl()


print(len(videos_annot_dct))
data_annot = []
labels = set()


for video_id, clips in videos_annot_dct.items():  # each video
    
    for clip_id, clip_data in clips.items():  # each clip
        category = clip_data['category']
        labels.add(category)
        
        for frame_id, boxes in clip_data["frame_boxes_dct"].items(): # Frames

            frame_path = f"{Paths.VIDEOS_ROOT.value}/{video_id}/{clip_id}/{frame_id}.jpg"
            data_annot.append(
                {
                "path": frame_path,
                "category" :category
                }
            )


print(f"{len(labels)} labels: {labels}\n\n")
for frame_dct in data_annot:
    for path, cat in frame_dct.items():
        print(frame_dct["path"], "==>", frame_dct["category"])
        break
    break

categories_dct = {
        'l-pass': 0,
        'r-pass': 1,
        'l-spike': 2,
        'r_spike': 3,
        'l_set': 4,
        'r_set': 5,
        'l_winpoint': 6,
        'r_winpoint': 7
    }

len_frames = len(data_annot)



# python -m src.DataLoaders.data_loader