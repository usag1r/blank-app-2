import os
import streamlit as st
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import csv
from PIL import Image
from torch.utils.data import Subset
import torch.nn as nn

from tqdm import tqdm
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.transforms as T

class TiledMNISTDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file, header=0)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        # Get the row
        img_name = self.labels_df.iloc[idx, 0]
        
        label1_str = self.labels_df.iloc[idx, 1]  # e.g. "10,6,6,8,2"
        label2_str = self.labels_df.iloc[idx, 2]  # e.g. "6,6,8,2,11"

        # Parse labels into list of ints
        label1 = torch.tensor([int(x) for x in label1_str.split(',')], dtype=torch.long)
        label2 = torch.tensor([int(x) for x in label2_str.split(',')], dtype=torch.long)

        # Load image
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image, label1, label2




transform = transforms.ToTensor()

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# transform = T.Compose([
#     T.Resize((56, 56)),    # make tiles bigger if needed
#     T.ToTensor(),          # convert to [1, H, W], range [0,1]
# ])

# ------------------------------------------------------
# Dataset: One digit per image
# ------------------------------------------------------
# transform = transforms.Compose([transforms.ToTensor()])



train_data = TiledMNISTDataset(
    csv_file='./data/imagetiles/tiled/2x2/train/labels.csv',
    image_dir='./data/imagetiles/tiled/2x2/train/',
    transform=transform
)

# Test a sample
img, label1, label2 = train_data[4]
print("Image shape:", img.shape)  # e.g., torch.Size([1, H, W])
print("Label 1:", label1)         # tensor([10, 6, 6, 8, 2])
print("Label 2:", label2)         # tensor([6, 6, 8, 2, 11])




test_data = TiledMNISTDataset(
    csv_file='data/imagetiles/tiled/2x2/test/labels.csv',
    image_dir='data/imagetiles/tiled/2x2/test/',
    transform=transform
)

# Test a sample
img_test, label1_test, label2_test = test_data[4]
print("Image shape:", img.shape)  # e.g., torch.Size([1, H, W])
print("Label 1:", label1_test)         # tensor([10, 6, 6, 8, 2])
print("Label 2:", label2_test)         # tensor([6, 6, 8, 2, 11])



#### Check the transform var
#### transform = transforms.Compose([transforms.ToTensor()])
#train_data = MNIST(root="./data/imagetiles/tiled/2x2/train", train=True, download=True, transform=transform)
#train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


# Select the indices you want (e.g. 0 through 9)
subset_indices = list(range(1000))

# Create the subset
train_subset = Subset(train_data, subset_indices)
test_subset = Subset(test_data, subset_indices)


# Use like a normal dataset
img, label1, label2 = train_subset[66]
print("Image shape:", img.shape)
print("Label 1:", label1)
print("Label 2:", label2)






# ---------------------
# Load Data
# ---------------------
transform = transforms.ToTensor()
data = TiledMNISTDataset(
    csv_file="data/imagetiles/tiled/2x2/train/labels.csv",
    image_dir="data/imagetiles/tiled/2x2/train/",
    transform=transform
)


# # ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

PATCH_SIZE = 14  # 56x56 MNIST / 14 = 4 patches per row/col → ? patches total
NUM_PATCHES = 16
EMBED_DIM = 256
NUM_CLASSES = 10
SEQ_LEN = 6  # max 4 digits + <start> + <end>
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

label_to_idx = {str(i): i for i in range(10)}
label_to_idx.update({"<start>": 10, "<end>": 11, "<pad>": 12})
idx_to_label = {v: k for k, v in label_to_idx.items()}
VOCAB_SIZE = len(label_to_idx)








































# ---------------------
# Streamlit UI
# ---------------------
st.title("Tiled MNIST Explorer")
st.markdown("Browse samples of your tiled MNIST dataset.")

idx = st.slider("Pick a sample index", 0, len(data) - 1, 0)
img, l1, l2, img_name = data[idx]

st.image(img.squeeze().numpy(), caption=f"{img_name}", use_column_width=True)
st.write("**Label 1 ⟶ Input sequence**:", l1.tolist())
st.write("**Label 2 ⟶ Target sequence**:", l2.tolist())

