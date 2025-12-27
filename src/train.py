import math
import random
import re
import string
from collections import Counter
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TinyBERT  # your TinyBERT implementation

# -------------------------------
# Load saved Punkt tokenizer and vocabulary
# -------------------------------
with open("punkt_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("vocab.pkl", "rb") as f:
    vocab_data = pickle.load(f)

vocab = vocab_data["vocab"]
word2idx = vocab_data["word2idx"]
idx2word = vocab_data["idx2word"]
max_seq_len = vocab_data["max_seq_len"]
vocab_size = len(vocab)

print("Vocab size:", vocab_size)
print("Max sequence length:", max_seq_len)

# -------------------------------
# Text preprocessing
# -------------------------------


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"'s\b", "", text)
    text = text.replace("\x0c", " ")
    text = re.sub(r"\(\s*\)", " ", text)
    text = re.sub(r"Part-\s*\w+", " ", text)
    text = re.sub(r"\b\d+\.\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


with open("constitution.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

clean_text = preprocess_text(raw_text)

# Tokenize using saved tokenizer
sentences = tokenizer.tokenize(clean_text)

# Further clean sentences
clean_sentences = []
for s in sentences:
    s = s.strip()
    if len(s) > 10 and not re.fullmatch(r"[^\w]+", s):
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\s+", " ", s).strip()
        clean_sentences.append(s)

print("Total sentences:", len(clean_sentences))

# -------------------------------
# Dataset for BERT-style masking
# -------------------------------


class MLMDataset(Dataset):
    def __init__(self, sentences, word2idx, seq_len, mask_prob=0.15):
        self.sentences = sentences
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.pad = word2idx["[PAD]"]
        self.cls = word2idx["[CLS]"]
        self.sep = word2idx["[SEP]"]
        self.mask = word2idx["[MASK]"]
        self.unk = word2idx["[UNK]"]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx].split()[: self.seq_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = [self.word2idx.get(t, self.unk) for t in tokens]
        labels = input_ids.copy()

        # BERT-style masking
        for i in range(1, len(input_ids) - 1):
            if random.random() < self.mask_prob:
                r = random.random()
                if r < 0.8:
                    input_ids[i] = self.mask  # 80% [MASK]
                elif r < 0.9:
                    input_ids[i] = random.randint(
                        5, vocab_size - 1)  # 10% random
                # else 10% keep original
            else:
                labels[i] = self.pad  # ignore token

        pad_len = self.seq_len - len(input_ids)
        input_ids += [self.pad] * pad_len
        labels += [self.pad] * pad_len
        attention_mask = [1 if t != self.pad else 0 for t in input_ids]

        return (
            torch.tensor(input_ids),
            torch.tensor(labels),
            torch.tensor(attention_mask),
        )


# -------------------------------
# Training
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MLMDataset(clean_sentences, word2idx, max_seq_len)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = TinyBERT(vocab_size, max_seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx["[PAD]"])

for epoch in range(100):
    total_loss = 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# -------------------------------
# Save model
# -------------------------------
model_path = "tinybert_model.pth"
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "vocab": vocab,
    "word2idx": word2idx,
    "idx2word": idx2word,
    "max_seq_len": max_seq_len
}, model_path)

print(f"Model saved to {model_path}")
