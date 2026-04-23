import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import gc
import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer

# Global variables
AA_VOCAB = list('ACDEFGHIKLMNPQRSTVWXY')
SS8_VOCAB = ['C', 'B', 'E', 'G', 'I', 'H', 'S', 'T']
NUM_CLASSES = len(SS8_VOCAB)
MAX_LEN = 700
REPLACEMENT_DICT = {
    ("A","V"), ("S","T"), ("F","Y"), ("K","R"), ("C","M"), ("D","E"), ("N","Q"), ("L","I"),
    ("V","A"), ("T","S"), ("Y","F"), ("R","K"), ("M","C"), ("E","D"), ("Q","N"), ("I","L")
} # no replacement for G, H, P, W, X
SAVE_DIR = "output/"

class ProteinDataset(Dataset):
    def __init__(self, df, embeddings, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings  # (N, 700, 320)
        self.ss_map = {c: i for i, c in enumerate(SS8_VOCAB)}
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def encode_labels(self, seq):
        arr = np.zeros(self.max_len, dtype=np.int64)
        for i, ch in enumerate(seq[:self.max_len]):
            if ch in self.ss_map:
                arr[i] = self.ss_map[ch]
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        X = self.embeddings[idx].astype(np.float32)  # (700, 320)
        y = self.encode_labels(row['dssp8'])          # (700,)
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



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False     # don't stop
        self.counter += 1
        return self.counter >= self.patience  # stop if patience exceeded


# https://link.springer.com/article/10.1186/s12859-025-06185-2#Sec2

class Adam_LM(Optimizer):
    def __init__(
            self,
            params,
            function,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            amsgrad=False,
            eps_lm=1,
    ):
        defaults = dict(
            function=function,
            c=0.,
            running_loss=0.,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            eps_lm=eps_lm,
        )
        super(Adam_LM, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_LM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            for group in self.param_groups:  # added this so that running_loss is actually updated
                group["running_loss"] = loss.item()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            beta1, beta2 = group["betas"]
            eps_lm = group["eps_lm"]
            func = group["function"]
            c = group["c"]
            running_loss = group["running_loss"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )

                    grad = p.grad
                    # new_grad = grad / (
                    #     func(torch.max(torch.zeros(p.data.size(), device=p.device), running_loss - c))
                    #     + eps_lm
                    # )
                    new_grad = grad / (
                            func(torch.clamp(torch.tensor(running_loss - c, device=p.device), min=0.0))
                            + eps_lm
                    )
                    grads.append(new_grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        ).to(p.device)
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        ).to(p.device)
                        if group["amsgrad"]:
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            ).to(p.device)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    state["step"] += 1

            self.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
            )

        return loss

    def adam(
            self,
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            *,
            amsgrad: bool,
            beta1: float,
            beta2: float,
            lr: float,
            weight_decay: float,
            eps: float,
    ):
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = self.state[param]["step"]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # denom = max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2).add_(eps) # doesn't work ??
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)) + eps
            else:
                # denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2).add_(eps)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)) + eps

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)




def train(data_loader, model, criterion, optimizer, device):
    train_avg_loss = 0
    num_correct = 0
    total_residues = 0

    model.train()  # Set model to training mode

    # Implement training loop
    for X, y, mask in tqdm(data_loader, leave=False):
        optimizer.zero_grad()
        X = X.to(device=device, dtype=torch.float32)
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
    print(f"Train loss: {final_loss} | Accuracy: {final_acc}")

    return final_loss, final_acc


def test(data_loader, model, criterion, device):
    test_avg_loss = 0
    num_correct = 0
    total_residues = 0

    model.eval()  # Set model to eval mode

    # Implement testing loop
    with torch.no_grad():
        for X, y, mask in tqdm(data_loader, leave=False):
            X = X.to(device=device, dtype=torch.float32)
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
    final_loss = test_avg_loss / len(data_loader)
    final_acc = num_correct / total_residues
    print(f"Test loss: {final_loss} | Accuracy: {final_acc}")

    return final_loss, final_acc


def run(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    BEST_MODEL_PATH = "output/best_model.pt"
    best_test_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
        test_loss, test_acc = test(test_dataloader, model, criterion, device)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        if early_stopping.step(test_loss): break

    model.load_state_dict(torch.load(BEST_MODEL_PATH))
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
    plt.show()
    plt.close()


def plot_confusion_matrix(data_loader, model, title, as_percent = True, class_names=SS8_VOCAB):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y, mask in tqdm(data_loader, leave=False):
            X = X.to(device=device, dtype=torch.float32)
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

    # save_path = DRIVE_PATH + "/Plots/LSTMwPLM/" + title + ".png"
    # fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    print(classification_report(all_labels, all_preds, target_names=class_names))
    plt.close()

    return cm


def experiment(
        train_loader,
        val_loader,
        # Hyperparameters
        weighted_loss=False,
        AA_size=len(AA_VOCAB),
        batch_size=32,
        dim_model=128,
        dropout=0.3,
        max_len=MAX_LEN,
        lr=1e-4,
        decay=1e-5,
        num_epochs=50,
        num_classes=NUM_CLASSES
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize model
    model = biGRU(dim_model=dim_model, dim_out=num_classes,
                         dropout=dropout)
    model.to(device)

    # Initialize loss function and optimizer
    if weighted_loss:
        # Calculate inverse frequency weights
        all_labels = ''.join(df_train['dssp8'].tolist())
        counts = Counter(all_labels)
        freqs = np.array([counts[c] for c in SS8_VOCAB], dtype=np.float32)
        weights = 1.0 / freqs
        weights = weights / weights.sum() * len(SS8_VOCAB)  # normalise

        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device), reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device=device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    # def identity(x):
    #     return x
    #
    # optim = Adam_LM(model.parameters(), identity, lr)

    # Run training and validation
    train_losses, train_accs, val_losses, val_accs = run(num_epochs=num_epochs, train_dataloader=train_loader,
                                                         test_dataloader=val_loader, model=model,
                                                         criterion=criterion, optimizer=optim, device=device)

    plot_name = f'_{"weighted" if weighted_loss else "unweighted"}'
    plot_over_epoch(train_losses, val_losses, "loss" + plot_name, "Loss")
    plot_over_epoch(train_accs, val_accs, "acc" + plot_name, "Accuracy")
    plot_confusion_matrix(val_loader, model, "cm" + plot_name)

    return train_losses, train_accs, val_losses, val_accs, model


train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

indices = np.arange(len(train_df))
train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)
train_idx_sorted = np.sort(train_idx)
val_idx_sorted = np.sort(val_idx)


df_train = train_df.iloc[train_idx_sorted].reset_index(drop=True)
df_val = train_df.iloc[val_idx_sorted].reset_index(drop=True)
del train_df
gc.collect()


train_embeddings = np.load("embeddings/train_embeddings.npy", mmap_mode='r')
test_embeddings = np.load("embeddings/test_embeddings.npy",  mmap_mode='r')

emb_train = train_embeddings[train_idx_sorted]
emb_val = train_embeddings[val_idx_sorted].copy()


train_dataset = ProteinDataset(df_train, emb_train)
val_dataset = ProteinDataset(df_val,   emb_val)
test_dataset = ProteinDataset(test_df,  test_embeddings)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=2)
val_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=2)


train_losses, train_accs, val_losses, val_accs, model = experiment(train_loader, val_loader, weighted_loss=False, dropout=0.5)

def eval_on_test(data_loader, model, class_names=SS8_VOCAB):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for X, y, mask in tqdm(data_loader, leave=False):
            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            mask = mask.to(device=device, dtype=torch.bool)

            y_hat = model.forward(X)
            pred = y_hat.argmax(dim=-1).to(dtype=torch.long)

            all_preds.append(pred[mask].cpu())
            all_labels.append(y[mask].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    print(classification_report(all_labels, all_preds, target_names=class_names))

eval_on_test(test_loader, model)