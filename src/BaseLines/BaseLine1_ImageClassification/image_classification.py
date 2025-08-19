"""
"""
from src.BaseLines.BaseLine1_ImageClassification.dataset_splitter import DatasetSplitter
from src.enums.PathEnums import Paths
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import os

# class ImageFolderCustom(Dataset):
#     def __init__(self, dir, labels, categories_dct, transform=None):
#         """
#         Initialize the dataset.
#         Args:
#             dir (str): The directory containing the images.
#             labels (list): The list of class labels.
#             categories_dct (dict): The dictionary mapping class labels to indices.
#             transform : A function/transform to apply to the images.
#         """

#         self.paths = self.get_list_of_files(dir)
#         # print(len(self.paths))
#         self.transform = transform
#         self.classes, self.class_to_idx = labels, categories_dct  # self.get_classes()
#         # self.labels = labels
#         # self.category_dict = categories_dct


#     def get_list_of_files(self, dir_path) -> list[str]:
#         """
#         Get a list of all image files in the directory.
#         Args:
#             dir_path (str): The directory path to search for image files.
#         Returns:
#             list[str]: A list of image file paths.
#         """
#         return list(Path(dir_path).glob("*/*.jpg"))
    
#     # def get_classes(self):

#     #     categories_dct = {
#     #     'l-pass': 0,
#     #     'r-pass': 1,
#     #     'l-spike': 2,
#     #     'r_spike': 3,
#     #     'l_set': 4,
#     #     'r_set': 5,
#     #     'l_winpoint': 6,
#     #     'r_winpoint': 7
#     #     }

#     #     labels = list(categories_dct.keys())

#     #     return labels, categories_dct

#     def load_image(self, index: int) -> Image:
#         """Load an image from a file path.
#         Args:
#             index (int): The index of the image to load.
#         Returns:
#             Image: The loaded image.
#         """
#         image_path = self.paths[index]
#         return Image.open(image_path)

#     def __len__(self) -> int:
#         "Returns the total number of samples."
#         return len(self.paths)

#     def __getitem__(self, index: int) -> tuple[Image, int]:
#         "Returns one sample of data, data and label (X, y)."
#         img = self.load_image(index)
#         class_name  = self.paths[index].parent.name # expects path ::>> data_folder/class_name/image.jpeg
#         class_idx = self.class_to_idx[class_name]

#         # Transform if necessary
#         if self.transform:
#             return self.transform(img), class_idx # return data, label (X, y)
#         else:
#             return (img, class_idx) # return data, label (X, y)

# =======================================================================================================


class CustomDataset(Dataset):
    def __init__(self, data_split, labels, transform=None):
        """
        Args:
            data_split: list of dicts [{"path": ..., "category": ...}, ...]
            labels: full list of class labels
            transform: torchvision transforms
        """
        self.data_split = data_split
        self.labels = labels
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(labels)}
        self.transform = transform

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        item = self.data_split[idx]
        # img = Image.open(item["path"]).convert("RGB")
        img = Image.open(item["path"]).convert("RGB")
        label = self.class_to_idx[item["category"]]
        if self.transform:
            img = self.transform(img)
        return img, label



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
    # ====================================================================

    print("transform")
    train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    train_dataset = CustomDataset(train_split, labels, transform=train_transforms)
    valid_dataset = CustomDataset(valid_split, labels, transform=test_transforms)
    test_dataset  = CustomDataset(test_split,  labels, transform=test_transforms)


    print(f"len train : {len(train_dataset)}")
    print(f"len valid : {len(valid_dataset)}")
    print(f"len test : {len(test_dataset)}")  
    #   # (26082, 13041, 4347)

    print(valid_dataset.labels)
    print(valid_dataset.class_to_idx)
    print("------------------------------------")

    image, label = train_dataset[0]

    class_names = list(train_dataset.labels)
    class_to_idx = train_dataset.class_to_idx
    print(f"classes : {class_names} \n classes idx : {class_to_idx}\n")
    print(f"image shape : {image.shape}, label:{label} => {class_names[label]}")


    # BATCH_SIZE = 32
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4)


    # print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}") #=> 816 || 32
    # print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")   #=> 136 || 32
    # print(f"Length of valid dataloader: {len(valid_dataloader)} batches of {BATCH_SIZE}") #=> 408 || 32


    # print("=========================")
    # for batch, i in enumerate (train_dataloader):  # c (Counter), i ==> [Batch([32, 1, 28, 28]), labels([32])]
    #     print("Batch", batch, "===> ", i[0].shape, i[1].shape)
    #     if batch == 5:
    #         break

    # train_features_batch, train_labels_batch = next(iter(train_dataloader))
    # print(train_features_batch.shape, train_labels_batch.shape)


# python -m src.BaseLines.BaseLine1_ImageClassification.image_classification
