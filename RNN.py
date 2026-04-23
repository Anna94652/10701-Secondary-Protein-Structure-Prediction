import math
import tqdm
import string
import time
import random
import numpy as np
import pandas as pd
import re, os, random
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)

AA_VOCAB  = list('ACDEFGHIKLMNPQRSTVWXY')
SS8_VOCAB = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
MAX_LEN   = 700

SAVE_PATH = "output/"
INPUT_PATH = ""

class ProteinDataset(Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df      = df.reset_index(drop=True)
        self.max_len = max_len
        self.aa_map  = {c: i for i, c in enumerate(AA_VOCAB)}
        self.ss_map  = {c: i for i, c in enumerate(SS8_VOCAB)}

    def __len__(self):
        return len(self.df)

    def encode(self, seq, vocab_map):
        arr = np.zeros(self.max_len, dtype=np.int64)
        for i, ch in enumerate(seq[:self.max_len]):
            if ch in vocab_map:
                arr[i] = vocab_map[ch]
        return arr

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        X      = self.encode(row['input'], self.aa_map)
        y      = self.encode(row['dssp8'], self.ss_map)
        length = min(len(row['input']), self.max_len)
        mask   = np.zeros(self.max_len, dtype=np.float32)
        mask[:length] = 1.0
        return (torch.tensor(X),
                torch.tensor(y),
                torch.tensor(mask))


class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, batch_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # Initialize embeddings, model layers, layer normalization, and activation function
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size, padding_idx=0)
        self.input_to_hidden = torch.nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.hidden_to_out = torch.nn.Linear(hidden_size, output_size)

        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()


    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # Implement forward pass including layer normalization
        h = hidden
        e = self.embedding(input)
        y = []
        for t in range(e.shape[1]):
            x_t = e[:, t, :]
            h = self.tanh(self.layer_norm(self.input_to_hidden(x_t) + self.hidden_to_hidden(h)))
            y_t = self.softmax(self.hidden_to_out(h))
            y.append(y_t)

        return torch.stack(y, dim=1)

    def init_hidden(self, batch_size, device):
      h0 = torch.zeros(batch_size, self.hidden_size, device=device)
      return h0


def train(data_loader, model, criterion, optimizer, device):
    model.train()

    total_loss = 0
    total_tokens = 0
    num_correct = 0

    for samples, labels, mask in tqdm.tqdm(data_loader, leave=False):
        samples = samples.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        B = samples.size(0)
        h0 = model.init_hidden(B, device)

        out = model(samples, h0)

        out_flat = out.view(-1, out.shape[-1])
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)

        loss = criterion(out_flat, labels_flat)
        loss = loss * mask_flat
        loss = loss.sum() / mask_flat.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask_flat.sum().item()
        total_tokens += mask_flat.sum().item()

        preds = torch.argmax(out, dim=-1)
        num_correct += ((preds == labels) * mask).sum().item()

    return total_loss / total_tokens, num_correct / total_tokens


# def test(data_loader, model, criterion, device):
#     total_loss = 0
#     total_tokens = 0
#     num_correct = 0

#     all_labels = []
#     all_preds = []

#     model.eval()

#     with torch.no_grad():
#         for samples, labels, mask in tqdm.tqdm(data_loader, leave=False):
#             samples = samples.to(device)
#             labels = labels.to(device)
#             mask = mask.to(device)

#             B = samples.size(0)
#             h0 = model.init_hidden(B, device)

#             out = model(samples, h0)

#             out_flat = out.view(-1, out.shape[-1])
#             labels_flat = labels.view(-1)
#             mask_flat = mask.view(-1)

#             loss = criterion(out_flat, labels_flat)
#             loss = loss * mask_flat
#             loss = loss.sum() / mask_flat.sum()

#             total_loss += loss.item() * mask_flat.sum().item()
#             total_tokens += mask_flat.sum().item()

#             preds = torch.argmax(out, dim=-1)
#             num_correct += ((preds == labels) * mask).sum().item()

#             # all_labels.append(labels)
#             # all_preds.append(preds)
#             mask = mask.bool()

#             labels = labels[mask]
#             preds = preds[mask]

#             all_labels.extend(labels.cpu().tolist())
#             all_preds.extend(preds.cpu().tolist())

#     return total_loss / total_tokens, num_correct / total_tokens, all_labels, all_preds


def test(data_loader, model, criterion, device):
    total_loss = 0
    total_tokens = 0
    num_correct = 0

    all_labels = []
    all_preds = []

    model.eval()

    with torch.no_grad():
        for samples, labels, mask in tqdm.tqdm(data_loader, leave=False):
            samples = samples.to(device)
            labels = labels.to(device)
            mask = mask.to(device).bool()

            B = samples.size(0)
            h0 = model.init_hidden(B, device)

            out = model(samples, h0)

            # flatten EVERYTHING first
            out_flat = out.view(-1, out.shape[-1])
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)

            loss = criterion(out_flat, labels_flat)
            loss = loss * mask_flat
            loss = loss.sum() / mask_flat.sum()

            total_loss += loss.item() * mask_flat.sum().item()
            total_tokens += mask_flat.sum().item()

            preds = torch.argmax(out, dim=-1)

            preds_flat = preds.view(-1)
            labels_flat2 = labels.view(-1)

            # safe masking
            valid = mask_flat

            all_labels.extend(labels_flat2[valid].cpu().tolist())
            all_preds.extend(preds_flat[valid].cpu().tolist())

            num_correct += (preds_flat[valid] == labels_flat2[valid]).sum().item()

    return (
        total_loss / total_tokens,
        num_correct / total_tokens,
        all_labels,
        all_preds
    )


def run(num_epochs, train_dataloader, test_dataloader, rnn, criterion, optimizer):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        # Perform one epoch of training and testing
        train_loss, train_acc = train(train_dataloader, rnn, criterion, optimizer, device)
        test_loss, test_acc, all_labels, all_preds = test(test_dataloader, rnn, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    return train_losses, test_losses, train_accs, test_accs, all_labels, all_preds # returns final test labels and preds


def plot_losses(epochs, train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train", color="blue")
    plt.plot(epochs, test_losses, label="Validation", color="red")
    plt.title("Loss Over Epochs (RNN)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    final_path = SAVE_PATH + 'RNN_losses_plot.png'
    plt.savefig(final_path)
    plt.close()


def plot_accs(epochs, train_accs, test_accs):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label="Train", color="blue")
    plt.plot(epochs, test_accs, label="Validation", color="red")
    plt.title("Accuracy Over Epochs (RNN)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    final_path = SAVE_PATH + 'RNN_accs_plot.png'
    plt.savefig(final_path)
    plt.close()


def plot_confusion_matrix(data_loader, model, title, as_percent=True, class_names=SS8_VOCAB):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y, mask in tqdm.tqdm(data_loader, leave=False):
            X = X.to(device=device, dtype=torch.long)
            y = y.to(device=device, dtype=torch.long)
            mask = mask.to(device=device, dtype=torch.bool)

            B = X.size(0)
            h0 = model.init_hidden(B, device)

            y_hat = model.forward(X, h0)
            pred = y_hat.argmax(dim=-1).to(dtype=torch.long)

            all_preds.append(pred[mask].cpu())
            all_labels.append(y[mask].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fig, ax = plt.subplots(figsize=(10, 8))

    # cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    cm = confusion_matrix(all_labels, all_preds)
    if as_percent:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", xticklabels=class_names, yticklabels=class_names,
                    ax=ax)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    final_path = SAVE_PATH + 'RNN_confusion_matrix.png'
    fig.savefig(final_path, bbox_inches='tight', dpi=150)
    # plt.show()
    # print(classification_report(all_labels, all_preds, target_names=class_names))
    plt.close()

    return cm


def main(
    # Hyperparameters
    batch_size = 128,
    input_size = 10,
    hidden_size = 64,
    max_review_length = MAX_LEN,
    lr = 1e-4,
    num_epochs = 50,
    num_classes = 2
):
    input_train = INPUT_PATH + 'train_data.csv'
    input_test = INPUT_PATH + 'test_data.csv'

    train_df = pd.read_csv(input_train)
    test_df = pd.read_csv(input_test)

    train_dataset = ProteinDataset(train_df)
    test_dataset = ProteinDataset(test_df)

    classes = SS8_VOCAB

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # input_size = MAX_LEN
    # num_train_samples = 10790
    # num_test_samples = 432
    AA_vocab_size = len(AA_VOCAB) # 21
    SS8_vocab_size = len(SS8_VOCAB) # 8
    num_classes = 8 # using SS8

    # Initialize model
    # model = RNN(AA_vocab_size, MAX_LEN, hidden_size, num_classes, batch_size)
    model = RNN(AA_vocab_size, input_size, hidden_size, num_classes)
    model.to(device)

    # Initialize loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optim = torch.optim.Adam(model.parameters(), lr)

    # Run training and testing
    train_losses, test_losses, train_accs, test_accs, all_labels, all_preds = run(num_epochs, train_loader, test_loader, model, loss_fn, optim)
    # run(num_epochs, train_dataloader, test_dataloader, model, loss_fn, optim)
    final_path = SAVE_PATH + 'rnn_model.pt'
    torch.save(model.state_dict(), final_path)

    # print(f"Train losses = {train_losses}")
    # print(f"Test losses = {test_losses}")

    plot_losses(list(range(num_epochs)), train_losses, test_losses)
    plot_accs(list(range(num_epochs)), train_accs, test_accs)
    plot_confusion_matrix(test_loader, model, "RNN Confusion Matrix", class_names=classes)
    return train_losses, test_losses

if __name__ == "__main__":
    main()