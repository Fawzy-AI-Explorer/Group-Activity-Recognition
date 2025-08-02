#!/bin/bash

echo -e "\n:package: Downloading Dataset in Progress..."

DATASET_LINK="https://www.kaggle.com/api/v1/datasets/download/ahmedmohamed365/volleyball"
DEST_DIR="data/volleyball.zip"
EX_DIR="data/volleyball"

# Check if zip file already exists
if [ -f "$DEST_DIR" ]; then
  echo -e "\n:warning:  Dataset already exists at $DEST_DIR"
  read -p "Do you want to re-download it? (y/n): " answer
  if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo -e "\n:fast_forward: Skipping download."
  else
    echo -e "\n:repeat: Re-downloading Dataset..."
    curl -L --create-dirs -o "$DEST_DIR" "$DATASET_LINK"
    echo -e "\n:white_check_mark: Downloading Dataset Completed Successfully"
  fi
else
  curl -L --create-dirs -o "$DEST_DIR" "$DATASET_LINK"
  echo -e "\n:white_check_mark: Downloading Dataset Completed Successfully"
fi

echo -e "\n:open_file_folder: Unzipping Files..."

unzip -o "$DEST_DIR" -d "$EX_DIR"

echo -e "\n:white_check_mark: Done Unzipping Files"

# zsh scripts/filename.sh