import os
import shutil
import random

# 📂 current data directories
DATASET_DIR = r"D:\py2\cataract_dataset"  # Kaynak
TRAIN_DIR = r"D:\py2\train"
TEST_DIR = r"D:\py2\test"

# clean up previous directories if they exist
# shutil.rmtree removes a directory and all its contents. It raises an error if the directory does not exist.
for folder in [TRAIN_DIR, TEST_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(os.path.join(folder, "immature"), exist_ok=True) # os.makedirs creates directories recursively, exist_ok = no error if exists.
    os.makedirs(os.path.join(folder, "mature"), exist_ok=True)

# operations for each category
for category in ["immature", "mature"]:
    src_folder = os.path.join(DATASET_DIR, category) # joins one or more path components intelligently. It constructs a full path by concatenating
    images = os.listdir(src_folder) # get the list of all files and directories

    # Random order
    random.shuffle(images)

    # %80 train, %20 test distribution
    split_idx = int(len(images) * 0.8) # train/test split index
    train_files = images[:split_idx] # %80 of imgs for training
    test_files = images[split_idx:] # rest %20 of imgs for testing

    # copy to train
    for img in train_files:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(TRAIN_DIR, category, img))

    # copy to test
    for img in test_files:
        shutil.copy(os.path.join(src_folder, img),
                    os.path.join(TEST_DIR, category, img))

print("✅data prepared successfully! Train/Test split done.")

# Note: This script assumes that the dataset is structured with two main folders: "immature" and "mature", each containing images.
