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

from model import build_model  # Import the function you wrapped in model.py

model = build_model()

print(model)


# ---------------------
# Streamlit UI
# ---------------------
st.title("Tiled MNIST Explorer")
st.markdown("Browse samples of your tiled MNIST dataset.")

###### Inference attempt with train data
#
######
# Assuming idx_to_label dict exists and maps indices to tokens like "0", "1", ..., "<start>", "<end>", "<pad>"

# model.eval()  # Set model to eval mode (important for dropout, batchnorm, etc.)

# with torch.no_grad():
#     # Get a small batch for inference (e.g., 32 samples)
#     sample_imgs, label1, label2 = next(iter(train_loader))
#     sample_imgs = sample_imgs.to(DEVICE)
#     label1 = label1.to(DEVICE)  # input sequence for decoder

#     # Forward pass: get logits (B, T, V)
#     logits = model(sample_imgs, label2)

#     # For example: get predicted token indices for each position in the sequence
#     pred_tokens = torch.argmax(logits, dim=-1)  # (B, T)
#     pred_digits = pred_tokens[:, :-1]           # (B, 4), ignore first token
#     print("Predicted token sequences (indices):")
#     print(pred_digits[:10].cpu())

#     # Now print actual target sequences from label2 (excluding <end> token if needed)
#     true_digits = label2[:, 1:]  # assuming label2 shape is (B, T), ignore <end> token to match length
#     true_digits = torch.cat([label1, label2[:, -1:]], dim=1)

#     print("\nActual target sequences (indices):")
#     print(true_digits[:10].cpu())

#     # Decode actual target sequences into strings
#     true_strings = []
#     for seq in true_digits[:10]:
#         true_strings.append(''.join([idx_to_label[idx.item()] for idx in seq]))

#     print("\nActual target sequences (decoded):")
#     for s in true_strings:
#         print(s)




# idx = st.slider("Pick a sample index", 0, len(data) - 1, 0)
# img, l1, l2, img_name = data[idx]

# st.image(img.squeeze().numpy(), caption=f"{img_name}", use_column_width=True)
# st.write("**Label 1 ⟶ Input sequence**:", l1.tolist())
# st.write("**Label 2 ⟶ Target sequence**:", l2.tolist())

