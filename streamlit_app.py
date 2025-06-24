import os
import torch
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# ---------------------
# Dataset Definition
# ---------------------
class TiledMNISTDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file, header=0)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        label1 = self.labels_df.iloc[idx, 1]
        label2 = self.labels_df.iloc[idx, 2]
        label1 = torch.tensor([int(x) for x in label1.split(',')], dtype=torch.long)
        label2 = torch.tensor([int(x) for x in label2.split(',')], dtype=torch.long)

        img = Image.open(os.path.join(self.image_dir, img_name)).convert("L")
        if self.transform:
            img = self.transform(img)

        return img, label1, label2, img_name

# ---------------------
# Load Data
# ---------------------
transform = transforms.ToTensor()
data = TiledMNISTDataset(
    csv_file="data/imagetiles/tiled/2x2/train/labels.csv",
    image_dir="data/imagetiles/tiled/2x2/train/",
    transform=transform
)

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
