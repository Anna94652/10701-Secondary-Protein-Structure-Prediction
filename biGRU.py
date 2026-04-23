import re, os, random
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import string
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Global variables
AA_VOCAB  = list('ACDEFGHIKLMNPQRSTVWXY')
SS8_VOCAB = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
NUM_CLASSES = len(SS8_VOCAB)
MAX_LEN = 700
REPLACEMENT_DICT = {
    ("A","V"), ("S","T"), ("F","Y"), ("K","R"), ("C","M"), ("D","E"), ("N","Q"), ("L","I"),
    ("V","A"), ("T","S"), ("Y","F"), ("R","K"), ("M","C"), ("E","D"), ("Q","N"), ("I","L")
} # no replacement for G, H, P, W, X


class ProteinDataset(Dataset):
    def __init__(self, df, p, aug_method, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.aa_map = {c: i for i, c in enumerate(AA_VOCAB)}
        self.ss_map = {c: i for i, c in enumerate(SS8_VOCAB)}
        self.p = p
        self.aug_method = aug_method

        self.lookup = np.arange(256, dtype=np.uint8)
        for src, dst in REPLACEMENT_DICT:
            self.lookup[ord(src)] = ord(dst)

    def __len__(self):
        return len(self.df)

    def summary(self):
        print("Length of sequeuences in dataset:")
        lengths = self.df['input'].str.len()
        print(lengths.describe())
        print(f"Max:    {lengths.max()}")
        print(f"Min:    {lengths.min()}")
        print(f"Median: {lengths.median()}")
        print(f"% over 700: {(lengths > 700).mean()*100:.1f}%")
        rows = {'length': lengths}
        for ss in self.ss_map.keys():
            rows[f'pct_{ss}'] = self.df["dssp8"].apply(
                lambda seq: seq.count(ss) / len(seq) if len(seq) > 0 else 0)
        print("Distrbution of labels:")
        summary = pd.DataFrame(rows)
        print(summary.describe())
        rows = {'length': lengths}
        for aa in self.aa_map.keys():
          rows[f'pct_{aa}'] = self.df['input'].apply(
              lambda seq: seq.count(aa) / len(seq) if len(seq) > 0 else 0)
        print("Distribution of inputs:")
        summary = pd.DataFrame(rows)
        print(summary.describe())

    def encode(self, seq, vocab_map):
        arr = np.zeros(self.max_len, dtype=np.int64)
        for i, ch in enumerate(seq[:self.max_len]):
            if ch in vocab_map:
                arr[i] = vocab_map[ch]
        return arr

    def replace_dictionary(self, sequence):
        mask = np.random.random(len(sequence)) < self.p
        replaced = self.lookup[sequence]
        return np.where(mask, replaced, sequence)

    def replace_alanine(self, sequence):
        mask = np.random.random(len(sequence)) < self.p
        alanine = np.full_like(sequence, ord('A'))
        return np.where(mask, alanine, sequence)

    def augment(self, seq_str):
        seq = np.frombuffer(seq_str.encode(), dtype=np.uint8).copy()
        if self.aug_method == 'dictionary':
            seq = self.replace_dictionary(seq)
        elif self.aug_method == 'alanine':
            seq = self.replace_alanine(seq)
        return ''.join(chr(c) for c in seq)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = self.augment(row['input']) if self.aug_method is not None else row['input']
        X = self.encode(seq, self.aa_map)
        y = self.encode(row['dssp8'], self.ss_map)
        length = min(len(row['input']), self.max_len)
        mask = np.zeros(self.max_len, dtype=np.float32)
        mask[:length] = 1.0
        return (torch.tensor(X),
                torch.tensor(y),
                torch.tensor(mask))


class GRU(nn.Module):
    def __init__(self, dim_embed, input_size, hidden_size, output_size, batch_size, sigma=0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embed_size = dim_embed
        self.output_size = output_size
        self.input_size = input_size
        self.sigma = sigma

        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (init_weight(input_size, hidden_size),
                          init_weight(hidden_size, hidden_size),
                          nn.Parameter(torch.zeros(hidden_size)))
        self.W_xz, self.W_hz, self.b_z = triple()  # Update gate
        self.W_xr, self.W_hr, self.b_r = triple()  # Reset gate
        self.W_xh, self.W_hh, self.b_h = triple()  # Candidate hidden state


    def forward(self, inputs, H=None):
      inputs = inputs.transpose(0, 1) # (T, B, D) - transposed to loop over time, not batch
      if H is None:
          # Initial state with shape: (batch_size, hidden_size)
          B = inputs.size(1)
          H = torch.zeros(B, self.hidden_size, device=inputs.device)

      outputs = []
      for X in inputs:
          assert X.shape[0] == H.shape[0], f"X:{X.shape}, H:{H.shape}"
          Z = torch.sigmoid(torch.matmul(X, self.W_xz) +
                          torch.matmul(H, self.W_hz) + self.b_z)
          R = torch.sigmoid(torch.matmul(X, self.W_xr) +
                          torch.matmul(H, self.W_hr) + self.b_r)
          H_tilde = torch.tanh(torch.matmul(X, self.W_xh) +
                            torch.matmul(R * H, self.W_hh) + self.b_h)
          H = Z * H + (1 - Z) * H_tilde
          outputs.append(H)

      outputs = torch.stack(outputs, dim=1)
      return outputs, H


# BIGRU
class biGRU(nn.Module):
    def __init__(self, input_size, dim_embed, dim_model, dim_out, batch_size, dropout=0.3):
        super().__init__()

        self.hidden_size = dim_model
        self.batch_size = batch_size
        self.embed_size = dim_embed
        self.output_size = dim_out
        self.input_size = input_size

        self.fwd_gru = GRU(dim_embed, input_size, dim_model, dim_out, batch_size) # Forward gru
        self.bwd_gru = GRU(dim_embed, input_size, dim_model, dim_out, batch_size) # Backward gru

        self.embedding = nn.Embedding(input_size, dim_embed)

    def forward(self, inputs, fwd_H=None, bwd_H=None):
        inputs = self.embedding(inputs)
        fwd_outputs, fwd_H = self.fwd_gru(inputs, fwd_H)

        bwd_inputs = torch.flip(inputs, dims=[1]) # reverse the inputs for backward gru
        bwd_outputs, bwd_H = self.bwd_gru(bwd_inputs, bwd_H)

        bwd_outputs2 = torch.flip(bwd_outputs, dims=[1]) # un-reverse the outputs from backward gru

        outputs = []
        for h_f, h_b in zip(fwd_outputs, bwd_outputs2):
            outputs.append(torch.cat([h_f, h_b], dim=1))


        outputs = torch.stack(outputs, dim=1)
        H = torch.cat([fwd_H, bwd_H], dim=-1)
        return outputs


def train(data_loader, model, criterion, optimizer, device):
    train_avg_loss = 0
    num_correct = 0
    total_residues = 0

    model.train()  # Set model to training mode

    # Implement training loop
    for X, y, mask in tqdm(data_loader, leave=False):
        print("Training i")
        optimizer.zero_grad()
        X = X.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.long)

        # Perform forward pass
        y_hat = model.forward(X)
        pred = y_hat.argmax(dim=-1).to(dtype=torch.long)

        # Calculate loss at every position and ignore padding
        loss = criterion(y_hat.reshape(-1, NUM_CLASSES), y.reshape(-1))
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()

        # Calculate number of correct predictions (ignore masked values)
        corr = torch.sum((y == pred) * mask)
        total_residues += mask.sum().item()

        # Backward pass, update weights, and zero gradients
        loss.backward()
        optimizer.step()

        num_correct += corr.item()
        train_avg_loss += loss.item()


    # print the loss and accuracy at the end of each epoch
    final_loss = train_avg_loss/len(data_loader)
    final_acc = num_correct/total_residues
    # print(f"Train loss: {final_loss} | Accuracy: {final_acc}")

    return final_loss, final_acc


def test(data_loader, model, criterion, device):
    test_avg_loss = 0
    num_correct = 0
    total_residues = 0

    model.eval() # Set model to eval mode

    # Implement testing loop
    with torch.no_grad():
      for X, y, mask in tqdm(data_loader, leave=False):
        X = X.to(device=device, dtype=torch.long)
        y = y.to(device=device, dtype=torch.long)
        mask = mask.to(device=device, dtype=torch.long)

        # Perform forward pass
        y_hat = model.forward(X)
        pred = y_hat.argmax(dim=-1).to(dtype=torch.long)

        # Calculate loss at every position and ignore padding
        loss = criterion(y_hat.reshape(-1, NUM_CLASSES), y.reshape(-1))
        loss = (loss * mask.reshape(-1)).sum() / mask.sum()

        # Calculate number of correct predictions (ignore masked values)
        corr = torch.sum((y == pred) * mask)
        total_residues += mask.sum().item()

        num_correct += corr.item()
        test_avg_loss += loss.item()

    # print the loss and accuracy at the end of each epoch
    final_loss = test_avg_loss/len(data_loader)
    final_acc = num_correct/total_residues
    # print(f"Test loss: {final_loss} | Accuracy: {final_acc}")

    return final_loss, final_acc


def run(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device):
  train_losses, test_losses = [], []
  train_accs, test_accs = [], []
  for epoch in range(num_epochs):
    # print(f"Epoch {epoch}")
    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
    test_loss, test_acc = test(test_dataloader, model, criterion, device)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

  return train_losses, train_accs, test_losses, test_accs


def plot_over_epoch(train_res, val_res, title, ylabel):
  epochs = range(1, len(train_res) + 1)
  plt.figure(figsize=(10, 6))

  plt.plot(epochs, train_res, label='Train', color='blue')
  plt.plot(epochs, val_res, label='Validation', color='red')

  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel(ylabel)
  plt.legend()
  plt.grid(True)

  save_path = "output/" + title + ".png"
  plt.savefig(save_path, bbox_inches='tight', dpi=150)
  # plt.show()
  plt.close()



def plot_confusion_matrix(data_loader, model, title, as_percent = True, class_names=SS8_VOCAB):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y, mask in tqdm(data_loader, leave=False):
            X = X.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.long)
            mask = mask.to(device=device, dtype=torch.bool)

            y_hat = model.forward(X)
            pred = y_hat.argmax(dim=-1).to(dtype=torch.long)

            all_preds.append(pred[mask].cpu())
            all_labels.append(y[mask].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    if as_percent:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    save_path = "output/" + title + ".png"
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    # plt.show()
    # print(classification_report(all_labels, all_preds, target_names=class_names))
    plt.close()

    return cm


def experiment(
    train_df,
    val_df,
    # Hyperparameters
    p = 0,
    aug_type = None,
    weighted_loss = False,
    AA_size = len(AA_VOCAB),
    batch_size = 32,
    dim_embed = 10,
    dim_model = 64,
    max_len = MAX_LEN,
    lr = 1e-4,
    num_epochs = 50,
    num_classes = NUM_CLASSES
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize model
    model = biGRU(input_size=AA_size, dim_embed=dim_embed, dim_model=dim_model, dim_out=num_classes, batch_size=batch_size)
    model.to(device)

    # Initialize data loaders
    train_dataset = ProteinDataset(train_df, p, aug_type)
    val_dataset  = ProteinDataset(val_df, 0, None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False)

    # Initialize loss function and optimizer
    if weighted_loss:
      # Calculate inverse frequency weights
      all_labels = ''.join(train_df['dssp8'].tolist())
      counts = Counter(all_labels)
      freqs   = np.array([counts[c] for c in SS8_VOCAB], dtype=np.float32)
      weights = 1.0 / freqs
      weights = weights / weights.sum() * len(SS8_VOCAB)  # normalise

      criterion = nn.CrossEntropyLoss(
          weight=torch.tensor(weights).to(device),
          reduction='none'
      )
    else:
      criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Run training and validation
    train_losses, train_accs, val_losses, val_accs = run(num_epochs=num_epochs, train_dataloader=train_loader,
                                                           test_dataloader=val_loader, model=model,
                                                            criterion=criterion, optimizer=optim, device=device)

    plot_name = f'_{"weighted" if weighted_loss else "unweighted"}_{"p0" if p == 0 else "p" + str(p)}_{"noAug" if not aug_type else aug_type}'
    plot_over_epoch(train_losses, val_losses, "loss" + plot_name, "Loss")
    plot_over_epoch(train_accs, val_accs, "acc" + plot_name, "Accuracy")
    plot_confusion_matrix(val_loader, model, "cm" + plot_name)

    return train_losses, train_accs, val_losses, val_accs, model



full_train_df = pd.read_csv("train_data.csv")
train_df, val_df = train_test_split(full_train_df, test_size=0.2, random_state=67)

ps = [i * 0.01 for i in range(11)]
aug_types = [None, "dictionary", "alanine"]
weights = [True, False]
results = []
for p in ps:
    for aug_type in aug_types:
        for weight in weights:
            print(f"Experiment - p = {p}, aug_type = {aug_type}, weighted = {weight}")
            train_losses, train_accs, val_losses, val_accs, model = experiment(
                train_df, val_df, p=p, aug_type=aug_type, weighted_loss=weight)
            results.append({
                'p': p,
                'aug_type': aug_type,
                'weighted_loss': weight,
                'train_loss': train_losses[-1],
                'train_acc': train_accs[-1],
                'val_loss': val_losses[-1],
                'val_acc': val_accs[-1],
            })

results_df = pd.DataFrame(results)
results_df.to_csv("output/GRU/grid_search_results.csv", index=False)