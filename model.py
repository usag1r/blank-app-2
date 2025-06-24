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


# ------------------------------------------------------
# Demo forward pass (training loop not included)
# ------------------------------------------------------
if __name__ == "__main__":
    model = ViTWithDecoder().to(DEVICE)

    # Get a batch from the loader
    sample_imgs, label1, label2 = next(iter(train_loader))
    sample_imgs = sample_imgs.to(DEVICE)
    label1 = label1.to(DEVICE)  # [B, 5] — starts with <start>
    label2 = label2.to(DEVICE)  # [B, 5] — ends with <end>

    # Pass through model
    logits = model(sample_imgs, label1)  # Pass label1 as target_seq

    print("Logits shape:", logits.shape)  # Expecting (B, T, V) = (batch_size, 5, vocab_size)
    train(model=model, train_loader=train_loader)



#train
#train(model=model, train_loader=train_loader)