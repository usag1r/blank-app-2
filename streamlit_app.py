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

from model import build_model  # Import the function you wrapped in model.py

model = build_model()



print(model)




























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

