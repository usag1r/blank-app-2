import streamlit as st
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os, csv
from PIL import Image
from torch.utils.data import Subset
import torch.nn as nn

from tqdm import tqdm
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_data = TiledMNISTDataset(
    csv_file='./data/imagetiles/tiled/2x2/train/labels.csv',
    image_dir='./data/imagetiles/tiled/2x2/train/',
    transform=transform
)


test_data = TiledMNISTDataset(
    csv_file='data/imagetiles/tiled/2x2/test/labels.csv',
    image_dir='data/imagetiles/tiled/2x2/test/',
    transform=transform
)

# Select the indices you want (e.g. 0 through 9)
subset_indices = list(range(100))

# Create the subset
train_subset = Subset(train_data, subset_indices)
test_subset = Subset(test_data, subset_indices)
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=True)

from model import build_model  # Import the function you wrapped in model.py

model = build_model()

##### Inference attempt with train data
#####Assuming idx_to_label dict exists and maps indices to tokens like "0", "1", ..., "<start>", "<end>", "<pad>"

label_to_idx = {str(i): i for i in range(10)}
label_to_idx.update({"<start>": 10, "<end>": 11, "<pad>": 12})
idx_to_label = {v: k for k, v in label_to_idx.items()}

model.eval()  # Set model to eval mode (important for dropout, batchnorm, etc.)



# ---------------------
# Streamlit UI
# ---------------------
st.title("Tiled MNIST Explorer")
st.markdown("##### © Senior Software Over-Engineering Pipeline Overlord")
st.markdown("Browse samples of your tiled MNIST dataset.")
st.write()

with torch.no_grad():
    # Get a small batch for inference (e.g., 32 samples)
    sample_imgs, label1, label2 = next(iter(train_loader))
    sample_imgs = sample_imgs.to(DEVICE)
    label1 = label1.to(DEVICE)  # input sequence for decoder

    # Forward pass: get logits (B, T, V)
    logits = model(sample_imgs, label2)

    # For example: get predicted token indices for each position in the sequence
    pred_tokens = torch.argmax(logits, dim=-1)  # (B, T)
    pred_digits = pred_tokens[:, :-1]           # (B, 4), ignore first token
    st.write("Predicted token sequences (indices):")
    st.write(pred_digits[:10].cpu())

    # Now print actual target sequences from label2 (excluding <end> token if needed)
    true_digits = label2[:, 1:]  # assuming label2 shape is (B, T), ignore <end> token to match length
    true_digits = torch.cat([label1, label2[:, -1:]], dim=1)

    st.write("\nActual target sequences (indices):")
    st.write(true_digits[:10].cpu())


    # Decode actual target sequences into strings
    true_strings = []
    for seq in true_digits[:10]:
        true_strings.append(''.join([idx_to_label[idx.item()] for idx in seq]))

    st.write("\nActual target sequences (decoded):")
    for s in true_strings:
        st.write(s)
