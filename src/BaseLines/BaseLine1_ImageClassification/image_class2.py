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
import logging

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


os.makedirs(ModelConfig.LOG_DIR.value, exist_ok=True)
os.makedirs(ModelConfig.LOG_CHECKPOINTS_DIR.value, exist_ok=True)
os.makedirs(ModelConfig.LOG_METRICSS_DIR.value, exist_ok=True)

class ResNet50Finetuner(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = True):
        """
        Fine-tuned ResNet50 model.

        Args:
            num_classes (int): number of classes in your dataset.
            freeze_backbone (bool): if True, freeze feature extractor and only train FC head.
            lr (float): learning rate for optimizer.
        """
        super().__init__()

        
        # Load pretrained ResNet50
        # self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        print("[INFO] : Loaded pretrained ResNet50... ")

        # Freeze backbone if required
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("[INFO] : Backbone frozen (only final FC will train)")
        else:
            print("[INFO] : All layers are trainable")

        # Replace final layer
        in_features = self.model.fc.in_features
        # self.model.fc = nn.Linear(in_features, num_classes)
        self.model.fc = nn.Sequential(
        # nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
        print(f"   Final layer replaced: {in_features} → {num_classes}")

        # Training components
        
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)

    def forward(self, x):
        return self.model(x)

    


def split_data():
    msg = "Start DatasetSplitter..."
    print(msg)
    logging.info(msg)

    splitter = DatasetSplitter()
    all_data, train_split, valid_split, test_split, labels = splitter.get_all_annotations()
    
    print("labels: ", labels, "\n")
    msg = f"len all data: {len(all_data)} || len train: {len(train_split)} || len valid: {len(valid_split)} || len test: {len(test_split)}"
    print(msg)
    logging.info(msg)

    print("==="*50, "\n")

    return train_split, valid_split, test_split, labels


def custom_data(train_split, valid_split, test_split, labels):
    msg = "Start CustomDataset..."
    print(msg)
    logging.info(msg)

   
    # train_transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    #     ], p=0.9),
    #     transforms.RandomHorizontalFlip(p=0.25),
    #     transforms.RandomVerticalFlip(p=0.25),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225])
    # ])

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ], p=0.9),

        # transforms.RandomHorizontalFlip(p=0.25),
        # transforms.RandomVerticalFlip(p=0.25),

        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
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

def data_loaders(train_dataset, valid_dataset, test_dataset):

  
    msg = "Strat DataLoader..."
    print(msg)
    logging.info(msg)

    BATCH_SIZE = ModelConfig.BATCH_SIZE.value

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dataloader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    msg = f"Length of train dataloader: {len(train_dataloader)} batches of {train_dataloader.batch_size}"
    print(msg)
    logging.info(msg)

    msg = f"Length of test dataloader: {len(test_dataloader)} batches of {test_dataloader.batch_size}"  
    print(msg)
    logging.info(msg)

    msg = f"Length of valid dataloader: {len(valid_dataloader)} batches of {valid_dataloader.batch_size}" 
    print(msg)
    logging.info(msg)

    # train_features_batch, train_labels_batch = next(iter(train_dataloader))
    # print(train_features_batch.shape, train_labels_batch.shape)

    return train_dataloader, valid_dataloader, test_dataloader

# ================

# ===============
def explore(model, device, input_size=(1, 3, 224, 224)):
        print("\n--- Model Summary ---\n")
        print(f"DEVICE:{device}")
        summary(model, input_size=input_size)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n   Total parameters: {total:,}")
        print(f"     Trainable parameters: {trainable:,}")
# =================

# =================
def print_train_time(model, device, start: float, end: float):
        total_time = end - start
        msg = f"Train time on {device}: {total_time:.3f} seconds"
        print(msg)
        logging.info(msg)
        return total_time
# ==============

# =============
def my_accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc
# ===============

# ===============
def overfit_on_batch(model, X_batch, y_batch, device, criterion, optimizer, acctorch, epochs=50):
        model.to(device)
        model.train()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        acctorch.reset()

        for epoch in range(epochs):
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            acctorch.update(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # acc = my_accuracy_fn(y_batch, y_pred.argmax(dim=1))
            acc = acctorch.compute().item()*100
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}, Acc: {acc}%")
# ===================

# ===================
def train_step(model, data_loader, device, criterion, acctorch, optimizer):
        model.to(device)
        model.train() # put model in train mode
        train_loss = 0.0
        acctorch.reset()

        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # Forward
            y_pred = model(X)
            loss = criterion(y_pred, y)

             # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            acctorch.update(y_pred, y)

            if batch%50==0:
                batch_acc = acctorch.compute().item() * 100
                msg = f"    Batch:{batch}/{len(data_loader)} | train loss: {loss.item():.4f} | train accuracy: {batch_acc:.2f}%"
                # print(msg)
                logging.info(msg)

    
        # Calculate loss and accuracy per epoch and print out what's happening
        avg_loss  = train_loss / len(data_loader)
        avg_acc = acctorch.compute()
        avg_acc = avg_acc.item()

        return avg_loss, avg_acc
# =================

# =================
def test_step(model, data_loader, device, criterion, acctorch, optimizer):
        model.to(device)
        model.eval() 
        test_loss = 0
        
        acctorch.reset()
        with torch.inference_mode():
            y_true, y_pred = [], []
            for batch, (X, y) in enumerate(data_loader):

                # Send data to GPU
                X, y = X.to(device), y.to(device)
                
                test_pred = model(X)
                loss = criterion(test_pred, y)


                test_loss += loss.item()
                acctorch.update(test_pred, y)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(test_pred.argmax(dim=1).cpu().numpy())
                # break  

                # if batch%100==0:
                #     batch_acc = acctorch.compute().item() * 100
                #     msg = f"    Batch:{batch}/{len(data_loader)} | valid loss: {loss.item():.4f} | valid accuracy: {batch_acc:.2f}%"
                #     print(msg)
                #     logging.info(msg)

            
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")
            report = classification_report(y_true, y_pred, output_dict=True)
            # print(cm, f1)
            avg_loss  = test_loss / len(data_loader)
            avg_acc = acctorch.compute()
            avg_acc = avg_acc.item()

            # print(f"Test loss: {avg_loss:.5f} | Test accuracy: {avg_acc}%\n")
            return avg_loss, avg_acc, cm, f1, report
# =================

# ================
def save_checkpoint(model, optimizer, save_path, epoch, best_val_acc):
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_acc": best_val_acc
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at {save_path}")
# ================

# ================
def predict(model, device, x: torch.Tensor):
        """Predict class for a single input tensor"""
        model.to(device)
        model.eval()
        with torch.inference_mode():
            x = x.unsqueeze(0).to(device)
            y_pred = model(x)
            return int(y_pred.argmax(dim=1).item())
# =================

# =================
def save_model(model, path="resnet50.pth"):
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

def load_model(model, device, path="resnet50.pth"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"Model loaded from {path}")


def load_checkpoint(model, optimizer, device, load_path):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_acc = checkpoint["best_val_acc"]
    print(f"Checkpoint loaded from {load_path}, resuming at epoch {start_epoch}")
    return start_epoch, best_val_acc
# ==============

# =============
def train_model(num_classes, train_dataloader, valid_dataloader, test_dataloader, lr, epochs, debug_overfit=False):
        logging.basicConfig(
            filename=os.path.join(ModelConfig.LOG_DIR.value, "training.log"),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.CrossEntropyLoss()
        acctorch = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        model = ResNet50Finetuner(num_classes=num_classes, freeze_backbone = False)
        # model.load_model(r"/teamspace/studios/this_studio/Group-Activity-Recognition/best_resnet50.pth")
        writer = SummaryWriter(log_dir=ModelConfig.RUNs_LOG_DIR.value)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-5) # 1e-3

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',       #  min val_loss 
                factor=0.1, patience=3)

        explore(model, device)

# =============================
# Overfit One Batch
# =============================
        if debug_overfit:
            # single batch
            msg = f"[INFO] : Overfit One Batch"
            logging.info(msg)
            X_batch, y_batch = next(iter(train_dataloader))
            overfit_on_batch(model, X_batch, y_batch, device, criterion, optimizer, acctorch, epochs=20)
            return model, {}
        
        msg = f"Start Training... "
        print(msg)
        logging.info(msg)
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        start_time = timer()
        best_val_acc = 0

        for epoch in range(1, epochs+1):

            print(f"Epoch: {epoch}/{epochs}")
            train_loss, train_acc = train_step(model, train_dataloader, device, criterion, acctorch, optimizer)


            val_loss, val_acc, cm, f1, report = test_step(model, valid_dataloader, device, criterion, acctorch, optimizer)
            scheduler.step(val_loss)
            # print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc*100:.2f}%")

            msg = f"    Train loss: {train_loss:.4f}, acc: {train_acc*100:.4f} | Val loss: {val_loss:.4f}, acc: {val_acc*100:.4f}"
            print(msg)
            logging.info(msg)

            pd.DataFrame(cm).to_csv(f"{ModelConfig.LOG_CF_MATRIX.value}{epoch}.csv", index=False)

            with open(f"{ModelConfig.LOG_CLS_REPORT.value}{epoch}.json", "w") as f:
                json.dump(report, f, indent=4)
            
            with open(ModelConfig.LOG_F1.value, "a") as f:
                f.write(f"Epoch {epoch}: {f1:.4f}\n")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, "best_resnet50.pth")

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val", f1, epoch)


            

            if epoch%5==0:
                save_checkpoint(model, optimizer, f"{ModelConfig.LOG_DIR.value}/checkpoints/{epoch}_resnet50.pth", epoch, best_val_acc)

        end_time = timer()
        print_train_time(model, device, start=start_time, end=end_time)

        with open(f"{ModelConfig.LOG_DIR.value}/training_history.json", "w") as f:
            json.dump(history, f, indent=4)

        pd.DataFrame(history).to_csv(f"{ModelConfig.LOG_DIR.value}/training_history.csv", index=False)
        print(f"Training history saved to {ModelConfig.LOG_DIR.value}/training_history.csv")


        test_loss, test_acc, cm, f1, report = test_step(model, test_dataloader, device, criterion, acctorch, optimizer)
        msg = f"Final Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%"
        print(msg)
        logging.info(msg)
        save_model(model, f"{ModelConfig.LOG_DIR.value}/final_resnet50.pth")

        return model, history
        # return history
# ================

# ===============
    


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

    

    model, history = train_model(num_classes, train_dataloader, valid_dataloader, test_dataloader, lr=ModelConfig.LR.value, epochs=ModelConfig.EPOCHS.value, debug_overfit=False)


    



if __name__ == "__main__":
    main()


# python -m src.BaseLines.BaseLine1_ImageClassification.image_class2
