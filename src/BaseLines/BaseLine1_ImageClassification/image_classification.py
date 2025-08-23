from src.BaseLines.BaseLine1_ImageClassification.dataset_splitter import DatasetSplitter
from src.BaseLines.BaseLine1_ImageClassification.custom_dataset import CustomDataset
from src.enums.PathEnums import Paths
from src.enums.ModelEnums import ModelConfig

from timeit import default_timer as timer
# from tqdm import tqdm
from pathlib import Path
import pandas as pd
from PIL import Image
import os
import json
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchmetrics.classification import Accuracy
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from torchinfo  import summary
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ResNet50Finetuner(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True, lr: float = 1e-3):
        """
        Fine-tuned ResNet50 model.

        Args:
            num_classes (int): number of classes in your dataset.
            freeze_backbone (bool): if True, freeze feature extractor and only train FC head.
            lr (float): learning rate for optimizer.
        """
        super().__init__()
        os.makedirs(ModelConfig.LOG_DIR.value, exist_ok=True)
        os.makedirs(ModelConfig.LOG_CHECKPOINTS_DIR.value, exist_ok=True)
        os.makedirs(ModelConfig.LOG_METRICSS_DIR.value, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.writer = SummaryWriter(log_dir=ModelConfig.RUNs_LOG_DIR.value)
        
        # Load pretrained ResNet50
        # self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        print("  Loaded pretrained ResNet50  ")

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("   Backbone frozen (only final FC will train)")
        else:
            print("   All layers are trainable")

        # Replace final layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        # for p in self.model.fc.parameters():
            # p.requires_grad = True
        print(f"   Final layer replaced: {in_features} → {num_classes}")

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.acctorch = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)


        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',       #  min val_loss 
            factor=0.1, 
            patience=3
        )

        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def explore(self, input_size=(1, 3, 224, 224)):
        """
        Print model summary and param counts.
        """
        print("\n--- Model Summary ---\n")
        print(f"DEVICE:{self.device}")

        summary(self.model, input_size=input_size)

        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n   Total parameters: {total:,}")
        print(f"     Trainable parameters: {trainable:,}")

    def print_train_time(self, start: float, end: float):
        total_time = end - start
        print(f"Train time on {self.device}: {total_time:.3f} seconds")
        return total_time
    
    def my_accuracy_fn(self, y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
    
    def overfit_on_batch(self, X_batch, y_batch, epochs=50):
        self.model.train()
        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

        for epoch in range(epochs):
            y_pred = self.model(X_batch)
            loss = self.criterion(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc = self.my_accuracy_fn(y_batch, y_pred.argmax(dim=1))
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Acc: {acc:.2f}%")


    def train_step(self,
               data_loader: torch.utils.data.DataLoader
                ):
        
        self.model.train() # put model in train mode
        train_loss, acc = 0.0, 0.0
        self.acctorch.reset()

        for batch, (X, y) in enumerate(data_loader):
            # if batch == 2:
            #     break
            # Send data to GPU
            X, y = X.to(self.device), y.to(self.device)
            

            # Forward
            y_pred = self.model(X)
            loss = self.criterion(y_pred, y)

             # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            train_loss += loss.item()
            # self.acctorch(y_pred.argmax(dim=1), y)
            self.acctorch.update(y_pred, y)

            # break

        
    
        # Calculate loss and accuracy per epoch and print out what's happening
        avg_loss  = train_loss / len(data_loader)
        avg_acc = self.acctorch.compute()
        avg_acc = avg_acc.item()
        # self.acctorch.reset()
        # print(f"Train loss: {avg_loss:.5f} | Train accuracy: {avg_acc*100}, || my_avg_acc")

        return avg_loss, avg_acc 


    def test_step(self,
                data_loader: torch.utils.data.DataLoader
                ):
        
        self.model.eval() 
        test_loss = 0
        
        with torch.inference_mode():
            y_true, y_pred = [], []
            # for X, y in data_loader:
            for batch, (X, y) in enumerate(data_loader):
                # if batch == 2:
                #     break
                # Send data to GPU
                X, y = X.to(self.device), y.to(self.device)
                
                test_pred = self.model(X)
                loss = self.criterion(test_pred, y)


                test_loss += loss.item()
                self.acctorch(test_pred.argmax(dim=1), y)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())
                # break
            
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            report = classification_report(y_true, y_pred, output_dict=True)
            # print(cm, f1)
            avg_loss  = test_loss / len(data_loader)
            avg_acc = self.acctorch.compute()
            avg_acc = avg_acc.item()
            self.acctorch.reset()
            # print(f"Test loss: {avg_loss:.5f} | Test accuracy: {avg_acc}%\n")
            return avg_loss, avg_acc, cm, f1, report

    def train_model(self, train_dataloader, valid_dataloader, epochs=5):
        """
        Train the model with optional validation.

        Args:
            train_loader: DataLoader for training data
            valid_loader: DataLoader for validation data
            epochs (int): number of training epochs
        """
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
    
        start_time = timer()
        best_val_acc = 0

        for epoch in range(1, epochs):

            # print(f"Epoch: {epoch}/{epochs}")
            train_loss, train_acc = self.train_step(train_dataloader)

            val_loss, val_acc, cm, f1, report = self.test_step(valid_dataloader)
            self.scheduler.step(val_loss)
            # print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc*100:.2f}%")

            pd.DataFrame(cm).to_csv(f"{ModelConfig.LOG_CF_MATRIX.value}{epoch}.csv", index=False)

            with open(f"{ModelConfig.LOG_CLS_REPORT.value}{epoch}.json", "w") as f:
                json.dump(report, f, indent=4)
            
            with open(ModelConfig.LOG_F1.value, "a") as f:
                f.write(f"Epoch {epoch}: {f1:.4f}\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model("best_resnet50.pth")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)

            print(f"Epoch {epoch}/{epochs} - "
                  f"Train loss: {train_loss:.4f}, acc: {train_acc*100:.4f}|"
                  f"Val loss: {val_loss:.4f}, acc: {val_acc*100:.4f}")
            if epoch%5==0:
                self.save_checkpoint(f"{ModelConfig.LOG_DIR.value}/checkpoints/{epoch}_resnet50", epoch, best_val_acc)

        end_time = timer()
        self.print_train_time(start=start_time, end=end_time)

        with open(f"{ModelConfig.LOG_DIR.value}/training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        pd.DataFrame(history).to_csv(f"{ModelConfig.LOG_DIR.value}/training_history.csv", index=False)
        print(f"Training history saved to {ModelConfig.LOG_DIR.value}/training_history.csv")
        return history

    def predict(self, x: torch.Tensor):
        """Predict class for a single input tensor"""
        self.model.eval()
        with torch.inference_mode():
            x = x.unsqueeze(0).to(self.device)
            y_pred = self.model(x)
            return int(y_pred.argmax(dim=1).item())

    def save_model(self, path="resnet50.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    def load_model(self, path="resnet50.pth"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        print(f"Model loaded from {path}")

    def save_checkpoint(self, save_path, epoch, best_val_acc):
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at {save_path}")

    def load_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        print(f"Checkpoint loaded from {load_path}, resuming at epoch {start_epoch}")
        return start_epoch, best_val_acc



def split_data():
    print("Strart DatasetSplitter...\n")

    splitter = DatasetSplitter(train_ratio=0.7, valid_ratio=0.1)
    all_data, train_split, valid_split, test_split, labels = splitter.split_dataset()
    
    # print("labels: ", labels, "\n")
    print(f"len data: {len(all_data)} || train: {len(train_split)} || valid: {len(valid_split)} || test: {len(test_split)}")
    print("==="*50, "\n")

    return train_split, valid_split, test_split, labels

def custom_data(train_split, valid_split, test_split, labels):
    print("Start CustomDataset...\n")

   
    # val_transforms = A.Compose([
    #     A.Resize(224, 224),
    #     A.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     ToTensorV2()
    # ])

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=0.9),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    # train_transforms = transforms.Compose([
    #         transforms.Resize((256, 256)),
    #         transforms.CenterCrop((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])

    test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop((224, 224)),
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
    # #   # (26082, 4347, 13041)

    # print(valid_dataset.labels)
    # print(valid_dataset.class_to_idx)
    print("="*50, "\n")
    return train_dataset, valid_dataset, test_dataset

def data_loaders(train_dataset, valid_dataset, test_dataset):

  
    print("Strat DataLoader...\n")
    BATCH_SIZE = ModelConfig.BATCH_SIZE.value
    epochs = ModelConfig.EPOCHS
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Length of train dataloader: {len(train_dataloader)} batches of {train_dataloader.batch_size}") #=> 816 || 32
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {test_dataloader.batch_size}")   #=> 136 || 32
    print(f"Length of valid dataloader: {len(valid_dataloader)} batches of {valid_dataloader.batch_size}") #=> 408 || 32

    # train_features_batch, train_labels_batch = next(iter(train_dataloader))
    # print(train_features_batch.shape, train_labels_batch.shape)

    return train_dataloader, valid_dataloader, test_dataloader

def train_model(num_classes, train_dataloader, valid_dataloader, test_dataloader):
    model = ResNet50Finetuner(num_classes=num_classes,
                             freeze_backbone = False, 
                             lr = ModelConfig.LR.value
                             )
    model.explore()

    # model.load_model(r"/teamspace/studios/this_studio/Group-Activity-Recognition/best_resnet50.pth")






    # X_batch, y_batch = next(iter(train_dataloader))
    # model.overfit_on_batch(X_batch, y_batch, epochs=30)

    history = model.train_model(
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        epochs=ModelConfig.EPOCHS.value
    )

    model.save_model(f"{ModelConfig.LOG_DIR.value}/final_resnet50.pth")



    test_loss, test_acc, cm, f1, report = model.test_step(test_dataloader)
    print(f"Final Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

    return model, history


def main():
    train_split, valid_split, test_split, labels = split_data()
    train_dataset, valid_dataset, test_dataset = custom_data(train_split, valid_split, test_split, labels)
    train_dataloader, valid_dataloader, test_dataloader = data_loaders(train_dataset, valid_dataset, test_dataset)
    
    # print(f"Length of train dataloader: {len(train_dataloader)} batches of {train_dataloader.batch_size}") #=> 816 || 32
    # print(f"Length of test dataloader: {len(test_dataloader)} batches of {test_dataloader.batch_size}")   #=> 136 || 32
    # print(f"Length of valid dataloader: {len(valid_dataloader)} batches of {valid_dataloader.batch_size}") #=> 408 || 32

    class_names = train_dataset.labels
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)
    print(class_to_idx)

    

    model, history = train_model(num_classes, train_dataloader, valid_dataloader, test_dataloader)

    



if __name__ == "__main__":
    main()


# python -m src.BaseLines.BaseLine1_ImageClassification.image_classification
