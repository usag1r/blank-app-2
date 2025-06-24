
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





# Creating a full MNIST ViT+Decoder pipeline as a .py file

label_to_idx = {str(i): i for i in range(10)}
label_to_idx.update({"<start>": 10, "<end>": 11, "<pad>": 12})
idx_to_label = {v: k for k, v in label_to_idx.items()}
VOCAB_SIZE = len(label_to_idx)

# ------------------------------------------------------
# Patch Embedder
# ------------------------------------------------------
class PatchEmbedder(nn.Module):
    def __init__(self, in_channels=1, embed_dim=EMBED_DIM, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, 28, 28) → (B, embed_dim, 4, 4) → (B, 16, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x  # (B, num_patches, embed_dim)


# ------------------------------------------------------
# Positional Encoding
# ------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches=NUM_PATCHES, dim=EMBED_DIM):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, x):
        return x + self.pos_embed


# ------------------------------------------------------
# Vision Transformer Encoder
# ------------------------------------------------------
class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, depth=4, heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

# ------------------------------------------------------
# AutoRegressive Decoder
# ------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_seq, memory):
        # tgt_seq: (B, T) → (B, T, D)
        tgt = self.token_embed(tgt_seq) + self.pos_embed[:, :tgt_seq.size(1)]
        tgt = tgt.transpose(0, 1)  # (T, B, D)
        memory = memory.transpose(0, 1)
        out = self.decoder(tgt, memory)
        return self.output_proj(out.transpose(0, 1))  # (B, T, V)

# ------------------------------------------------------
# Full ViT + Decoder Model
# ------------------------------------------------------
class ViTWithDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedder()
        self.pos_encoder = PositionalEncoding()
        self.encoder = ViTEncoder()
        self.decoder = Decoder()

    def forward(self, img, tgt_seq):
        x = self.patch_embed(img)                  # (B, 16, D)
        x = self.pos_encoder(x)                    # (B, 16, D)
        encoded = self.encoder(x)                  # (B, 16, D)
        logits = self.decoder(tgt_seq, encoded)    # (B, T, V)
        return logits



class ViTWithDecoderAndClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbedder()
        self.pos_encoder = PositionalEncoding()
        self.encoder = ViTEncoder()
        self.decoder = Decoder()
        self.classifier = nn.Linear(EMBED_DIM, NUM_CLASSES)  # e.g., NUM_CLASSES=10 for MNIST digits

    def forward(self, img, tgt_seq):
        x = self.patch_embed(img)          # (B, num_patches, D)
        x = self.pos_encoder(x)            # (B, num_patches, D)
        encoded = self.encoder(x)          # (B, num_patches, D)

        logits = self.decoder(tgt_seq, encoded)  # decoder output logits (B, T, V)

        # For classification, typically pool encoder outputs
        # e.g., mean pooling over patches:
        pooled = encoded.mean(dim=1)       # (B, D)
        class_logits = self.classifier(pooled)  # (B, NUM_CLASSES)

        return logits, class_logits


import torch.optim as optim
import torch.nn.functional as F

def train(model, train_loader, epochs=1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=label_to_idx["<pad>"])

    for epoch in range(epochs):
        total_loss = 0
        for imgs, label1, label2 in train_loader:
            imgs = imgs.to(DEVICE)
            label1 = label1.to(DEVICE)  # input seq, e.g. [<start>, digits...]
            label2 = label2.to(DEVICE)  # target seq, e.g. [digits..., <end>]

            optimizer.zero_grad()
            logits = model(imgs, label1)  # logits: (B, T, V)
            
            # Reshape for loss: (B*T, V) and (B*T) targets
            B, T, V = logits.shape
            loss = criterion(logits.view(B*T, V), label2.view(B*T))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", end=" ||| ")



def build_model():
    model = ViTWithDecoder().to(DEVICE)
    model = train(model=model, train_loader=train_loader)
    return model

def run_demo():
    model = build_model()
    sample_imgs, label1, label2 = next(iter(train_loader))
    sample_imgs = sample_imgs.to(DEVICE)
    label1 = label1.to(DEVICE)
    label2 = label2.to(DEVICE)
    logits = model(sample_imgs, label1)
    print("Logits shape:", logits.shape)
    train(model)

if __name__ == "__main__":
    run_demo()


#train
#train(model=model, train_loader=train_loader)