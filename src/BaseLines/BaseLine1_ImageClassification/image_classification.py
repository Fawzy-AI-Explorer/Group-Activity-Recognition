"""
"""

from src.BaseLines.BaseLine1_ImageClassification.dataset_splitter import DatasetSplitter
from src.enums.PathEnums import Paths
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image


class ImageFolderCustom(Dataset):
    def __init__(self, dir, labels, categories_dct, transform=None):
        """
        Initialize the dataset.
        Args:
            dir (str): The directory containing the images.
            labels (list): The list of class labels.
            categories_dct (dict): The dictionary mapping class labels to indices.
            transform : A function/transform to apply to the images.
        """

        self.paths = self.get_list_of_files(dir)
        # print(len(self.paths))
        self.transform = transform
        self.classes, self.class_to_idx = labels, categories_dct  # self.get_classes()
        # self.labels = labels
        # self.category_dict = categories_dct


    def get_list_of_files(self, dir_path) -> list[str]:
        """
        Get a list of all image files in the directory.
        Args:
            dir_path (str): The directory path to search for image files.
        Returns:
            list[str]: A list of image file paths.
        """
        return list(Path(dir_path).glob("*/*.jpg"))
    
    # def get_classes(self):

    #     categories_dct = {
    #     'l-pass': 0,
    #     'r-pass': 1,
    #     'l-spike': 2,
    #     'r_spike': 3,
    #     'l_set': 4,
    #     'r_set': 5,
    #     'l_winpoint': 6,
    #     'r_winpoint': 7
    #     }

    #     labels = list(categories_dct.keys())

    #     return labels, categories_dct

    def load_image(self, index: int) -> Image:
        """Load an image from a file path.
        Args:
            index (int): The index of the image to load.
        Returns:
            Image: The loaded image.
        """
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Image, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path ::>> data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return (img, class_idx) # return data, label (X, y)



if __name__ == "__main__":
    annot_loader = DatasetSplitter()
    data_annot, labels, categories_dct = annot_loader.build_annotations()

    # train_data, valid_data, test_data = annot_loader.split(data_annot)
    # # print(int(len(data_annot)*0.6) +  int(len(data_annot)*0.3) + int(len(data_annot)*0.1)) # 43470
    # print("Train...") # 25973
    # annot_loader.save_split(train_data, Paths.TRAIN_PATH.value)
    # print("Valid...") # 12861
    # annot_loader.save_split(valid_data, Paths.VALID_PATH.value)
    # print("Test...") # 4300
    # annot_loader.save_split(test_data, Paths.TEST_PATH.value)
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


    train_data_custom = ImageFolderCustom(dir=Paths.TRAIN_PATH.value,
                                          labels=labels, categories_dct=categories_dct,
                                          transform=train_transforms
                                          )

    valid_data_custom = ImageFolderCustom(dir=Paths.VALID_PATH.value,
                                          labels=labels, 
                                          categories_dct=categories_dct,
                                          transform=test_transforms
                                          )

    test_data_custom = ImageFolderCustom(dir=Paths.TEST_PATH.value, 
                                         labels=labels, 
                                         categories_dct=categories_dct,
                                         transform=test_transforms
                                         )

    print(f"len train : {len(train_data_custom)}")
    print(f"len valid : {len(valid_data_custom)}")
    print(f"len test : {len(test_data_custom)}")  
      # (26082, 13041, 4347)

    image, label = train_data_custom[0]
    class_names = train_data_custom.classes
    class_to_idx = train_data_custom.class_to_idx
    print(class_names, class_to_idx)
    print(f"classes : {class_names} \n classes idx : {class_to_idx}\n")
    print(f"image shape : {image.shape}")


    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_data_custom, 
                                     batch_size=BATCH_SIZE, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                        batch_size=BATCH_SIZE, 
                                        num_workers=0, 
                                        shuffle=False) # don't usually need to shuffle testing data

    valid_dataloader = DataLoader(dataset=valid_data_custom, # use custom created valid Dataset
                                         batch_size=BATCH_SIZE, 
                                         num_workers=0,
                                         shuffle=False) # don't usually need to shuffle validation data

    print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}") #=> 816 || 32
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")   #=> 136 || 32
    print(f"Length of valid dataloader: {len(valid_dataloader)} batches of {BATCH_SIZE}") #=> 408 || 32


    print("=========================")
    for batch, i in enumerate (train_dataloader):  # c (Counter), i ==> [Batch([32, 1, 28, 28]), labels([32])]
        print("Batch", batch, "===> ", i[0].shape, i[1].shape)
        if batch == 5:
            break

    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(train_features_batch.shape, train_labels_batch.shape)


# python -m src.BaseLines.BaseLine1_ImageClassification.image_classification
