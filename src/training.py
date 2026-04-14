"""Training, evaluation, and checkpoint utilities for multi-label GNN."""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss


def kaiming_init(model):
    """Kaiming uniform initialization for ReLU-based networks.

    Kaiming compensates for ReLU zeroing ~50% of activations, giving ~41%
    larger initial weights than Xavier for the same layer size.  This improves
    convergence on weak-signal data (e.g. feature-zeroed unlearning graphs).
    """
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            nn.init.kaiming_uniform_(param, nonlinearity='relu')
        elif 'bias' in name:
            nn.init.zeros_(param)


def get_pos_weights(y_train, device):
    """Compute pos_weight for BCEWithLogitsLoss: neg_count / pos_count per label."""
    y = np.array(y_train)
    pos = y.sum(axis=0)
    neg = len(y) - pos
    pw = neg / np.maximum(pos, 1)
    return torch.tensor(pw, dtype=torch.float).to(device)


def train_model(model, train_loader, val_loader, device,
                epochs=200, lr=5e-4, weight_decay=1e-4, patience=30,
                pos_weights=None, verbose=False, scheduler_patience=20):
    """Train with BCEWithLogitsLoss + ReduceLROnPlateau + early stopping.

    Monitors macro-averaged ROC-AUC across all EVCS labels.
    Returns (model, best_val_metric, epoch_logs).
    epoch_logs is a list of dicts: [{epoch, train_loss, val_roc}, ...].
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    best_metric = 0.0
    best_state = None
    wait = 0
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        # --- Validate ---
        val_metric = _compute_val_roc(model, val_loader, device)
        scheduler.step(val_metric)

        epoch_logs.append({
            'epoch': epoch,
            'train_loss': round(avg_loss, 6),
            'val_roc': round(val_metric, 6),
        })

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}, best val macro-ROC={best_metric:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_metric, epoch_logs


def _compute_val_roc(model, val_loader, device):
    """Compute macro ROC-AUC on validation set."""
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            all_logits.append(model(batch).cpu())
            all_y.append(batch.y.cpu())
    logits = torch.cat(all_logits).numpy()
    ys = torch.cat(all_y).numpy()
    probs = 1 / (1 + np.exp(-logits))

    if np.any(np.isnan(probs)):
        return 0.5
    valid_cols = [c for c in range(ys.shape[1]) if len(np.unique(ys[:, c])) > 1]
    if len(valid_cols) == 0:
        return 0.5
    return float(np.mean([roc_auc_score(ys[:, c], probs[:, c]) for c in valid_cols]))


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Multi-label evaluation.

    Returns dict with: exact_match, hamming_acc, macro_f1, macro_roc,
                       per_roc, per_f1, preds, probs, labels.
    """
    model.eval()
    all_logits, all_y = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch).cpu())
        all_y.append(batch.y.cpu())

    logits = torch.cat(all_logits).numpy()
    ys = torch.cat(all_y).numpy()
    probs = 1 / (1 + np.exp(-logits))
    if np.any(np.isnan(probs)):
        probs = np.nan_to_num(probs, nan=0.5)

    preds = (probs >= 0.5).astype(float)
    exact_match = (preds == ys).all(axis=1).mean()
    h_acc = 1.0 - hamming_loss(ys, preds)

    n_labels = ys.shape[1]
    per_roc, per_f1 = [], []
    for c in range(n_labels):
        if len(np.unique(ys[:, c])) > 1:
            per_roc.append(roc_auc_score(ys[:, c], probs[:, c]))
            per_f1.append(f1_score(ys[:, c], preds[:, c]))
        else:
            per_roc.append(0.5)
            per_f1.append(0.0)

    return {
        'exact_match': exact_match,
        'hamming_acc': h_acc,
        'macro_f1': float(np.mean(per_f1)),
        'macro_roc': float(np.mean(per_roc)),
        'per_roc': per_roc,
        'per_f1': per_f1,
        'preds': preds,
        'probs': probs,
        'labels': ys,
    }


def compute_mia_auc(model, member_loader, non_member_loader, device,
                    pos_weights=None):
    """Loss-based MIA for multi-label. AUC near 0.5 = good forgetting."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
    losses, labels = [], []

    with torch.no_grad():
        for batch in member_loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch.y).mean(dim=1)
            losses.append(loss.cpu())
            labels.append(torch.ones(len(loss)))

        for batch in non_member_loader:
            batch = batch.to(device)
            loss = criterion(model(batch), batch.y).mean(dim=1)
            losses.append(loss.cpu())
            labels.append(torch.zeros(len(loss)))

    losses = torch.cat(losses).numpy()
    labels = torch.cat(labels).numpy()

    if np.any(np.isnan(losses)):
        return 0.5
    scores = -losses
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def save_checkpoint(model, path, backbone, scenario, seed):
    """Save model state_dict to results/checkpoints/."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filename = os.path.join(path, f"{backbone}_{scenario}_{seed}.pth")
    torch.save(model.state_dict(), filename)
    return filename
