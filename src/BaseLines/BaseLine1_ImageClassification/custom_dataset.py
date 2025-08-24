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
        img = Image.open(item["path"]).convert("RGB")
        label = self.class_to_idx[item["category"]]

        if self.transform:
            img = self.transform(img)

        return img, label
# ==============================================================================

def custom_data(train_split, valid_split, test_split, labels):
    print("Start CustomDataset...\n")


    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        # transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ], p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    test_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
                )
        ])

    train_dataset = CustomDataset(train_split, labels, transform=train_transforms)
    valid_dataset = CustomDataset(valid_split, labels, transform=test_transforms)
    test_dataset  = CustomDataset(test_split,  labels, transform=test_transforms)

    print(f"len train : {len(train_dataset)}")
    print(f"len valid : {len(valid_dataset)}")
    print(f"len test : {len(test_dataset)}")  

    print(valid_dataset.labels)
    print(valid_dataset.class_to_idx)
    print("="*50, "\n")
    return train_dataset, valid_dataset, test_dataset
# =======================================================================

if __name__ == "__main__":

    splitter = DatasetSplitter()
    all_data, train_split, valid_split, test_split, labels = splitter.get_all_annotations()
    print("\n==================================================================================\n")
    train_dataset, valid_dataset, test_dataset = custom_data(train_split, valid_split, test_split, labels)
    print("\n==================================================================================\n")
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4)


# python -m src.BaseLines.BaseLine1_ImageClassification.custom_dataset
