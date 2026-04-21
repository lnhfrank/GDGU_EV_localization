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
                    pos_weights=None, forget_label_idx=None):
    """Loss-based MIA for multi-label, split by forget/retain labels.

    Following OpenGU (Fan et al., 2025), MIA should measure information
    leakage about the *forgotten* data specifically, not the overall
    train/test generalization gap.

    Args:
        forget_label_idx: list of label column indices for forget EVCS.
            e.g. S1 forgets EVCS1 -> [0]; S2 -> [0,1]; S3 -> [0,1,2].
            If None, falls back to overall-only (backward compatible).

    Returns:
        dict with keys: mia_forget, mia_retain, mia_overall.
        mia_retain is NaN when all labels are forget (e.g. S3).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
    all_losses, all_labels = [], []  # per-label losses [B, n_labels]

    with torch.no_grad():
        for loader, lbl in [(member_loader, 1), (non_member_loader, 0)]:
            for batch in loader:
                batch = batch.to(device)
                loss = criterion(model(batch), batch.y)  # [B, n_labels]
                all_losses.append(loss.cpu())
                all_labels.append(torch.full((loss.size(0),), lbl))

    losses = torch.cat(all_losses).numpy()   # [N, n_labels]
    labels = torch.cat(all_labels).numpy()   # [N]

    if np.any(np.isnan(losses)) or len(np.unique(labels)) < 2:
        return {'mia_forget': 0.5, 'mia_retain': 0.5, 'mia_overall': 0.5}

    def _auc(col_idx):
        if not col_idx:
            return np.nan
        scores = -losses[:, col_idx].mean(axis=1)
        return float(roc_auc_score(labels, scores))

    n_labels = losses.shape[1]
    all_idx = list(range(n_labels))

    if forget_label_idx is not None:
        retain_idx = [i for i in all_idx if i not in forget_label_idx]
        mia_forget = _auc(forget_label_idx)
        mia_retain = _auc(retain_idx)
    else:
        mia_forget = np.nan
        mia_retain = np.nan

    mia_overall = _auc(all_idx)
    return {'mia_forget': mia_forget, 'mia_retain': mia_retain,
            'mia_overall': mia_overall}


def save_checkpoint(model, path, backbone, scenario, seed):
    """Save model state_dict to results/checkpoints/."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filename = os.path.join(path, f"{backbone}_{scenario}_{seed}.pth")
    torch.save(model.state_dict(), filename)
    return filename


# ======================================================================
#  Dual-Channel training / evaluation / MIA
# ======================================================================
#
# The dual-channel model returns (loc_logits [B, K], det_logits [B]).
# Loss = alpha * BCE(det) + beta * BCE(loc).  Strict isolation: the two
# heads share NO parameters, so there is no cross-contamination.
# ======================================================================


def train_model_dual(model, train_loader, val_loader, device,
                     epochs=200, lr=5e-4, weight_decay=1e-4, patience=30,
                     pos_weights=None, alpha=1.0, beta=1.0,
                     verbose=False, scheduler_patience=20,
                     only_params=None):
    """Multi-task training for DualChannel_Graph.

    Monitors val macro-ROC on localization (primary task) for early stopping.

    Args:
        alpha: detection loss weight.
        beta:  localization loss weight.
        only_params: if given, only these params are optimized (used in
            fine-tune after GDGU when graph channel should remain frozen).
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    params = only_params if only_params is not None \
        else [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience)
    crit_loc = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    crit_det = nn.BCEWithLogitsLoss()

    best_metric = 0.0
    best_state = None
    wait = 0
    epoch_logs = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_loc = 0.0
        total_det = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loc_logits, det_logits = model(batch)
            y_det = batch.y_det.view(-1)
            loss_loc = crit_loc(loc_logits, batch.y)
            loss_det = crit_det(det_logits, y_det)
            loss = alpha * loss_det + beta * loss_loc
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()
            total_loc += loss_loc.item()
            total_det += loss_det.item()
            n_batches += 1
        n_batches = max(n_batches, 1)

        val_metric = _compute_val_roc_dual(model, val_loader, device)
        scheduler.step(val_metric)

        epoch_logs.append({
            'epoch': epoch,
            'train_loss': round(total_loss / n_batches, 6),
            'train_loss_loc': round(total_loc / n_batches, 6),
            'train_loss_det': round(total_det / n_batches, 6),
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


def _compute_val_roc_dual(model, val_loader, device):
    """Val macro ROC-AUC on localization logits (primary task)."""
    model.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            loc_logits, _ = model(batch)
            all_logits.append(loc_logits.cpu())
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
def evaluate_model_dual(model, loader, device):
    """Multi-task evaluation: localization metrics + detection accuracy.

    Returns dict with localization fields (same as evaluate_model) plus:
        det_acc, det_auc, det_f1.
    """
    from sklearn.metrics import accuracy_score
    model.eval()
    loc_logits_all, det_logits_all = [], []
    y_loc_all, y_det_all = [], []
    for batch in loader:
        batch = batch.to(device)
        loc_logits, det_logits = model(batch)
        loc_logits_all.append(loc_logits.cpu())
        det_logits_all.append(det_logits.cpu())
        y_loc_all.append(batch.y.cpu())
        y_det_all.append(batch.y_det.view(-1).cpu())

    loc_logits = torch.cat(loc_logits_all).numpy()
    det_logits = torch.cat(det_logits_all).numpy()
    y_loc = torch.cat(y_loc_all).numpy()
    y_det = torch.cat(y_det_all).numpy()

    loc_probs = 1 / (1 + np.exp(-loc_logits))
    det_probs = 1 / (1 + np.exp(-det_logits))
    if np.any(np.isnan(loc_probs)):
        loc_probs = np.nan_to_num(loc_probs, nan=0.5)
    if np.any(np.isnan(det_probs)):
        det_probs = np.nan_to_num(det_probs, nan=0.5)

    loc_preds = (loc_probs >= 0.5).astype(float)
    det_preds = (det_probs >= 0.5).astype(float)

    exact_match = (loc_preds == y_loc).all(axis=1).mean()
    h_acc = 1.0 - hamming_loss(y_loc, loc_preds)

    n_labels = y_loc.shape[1]
    per_roc, per_f1 = [], []
    for c in range(n_labels):
        if len(np.unique(y_loc[:, c])) > 1:
            per_roc.append(roc_auc_score(y_loc[:, c], loc_probs[:, c]))
            per_f1.append(f1_score(y_loc[:, c], loc_preds[:, c]))
        else:
            per_roc.append(0.5)
            per_f1.append(0.0)

    det_acc = float(accuracy_score(y_det, det_preds))
    det_auc = float(roc_auc_score(y_det, det_probs)) if len(np.unique(y_det)) > 1 else 0.5
    det_f1 = float(f1_score(y_det, det_preds)) if len(np.unique(y_det)) > 1 else 0.0

    return {
        'exact_match': exact_match,
        'hamming_acc': h_acc,
        'macro_f1': float(np.mean(per_f1)),
        'macro_roc': float(np.mean(per_roc)),
        'per_roc': per_roc,
        'per_f1': per_f1,
        'preds': loc_preds,
        'probs': loc_probs,
        'labels': y_loc,
        'det_acc': det_acc,
        'det_auc': det_auc,
        'det_f1': det_f1,
    }


def compute_mia_auc_dual(model, member_loader, non_member_loader, device,
                         pos_weights=None, forget_label_idx=None):
    """Loss-based MIA for dual-channel model using ONLY localization loss.

    Rationale: the detection head is a privacy-neutral signal (grid operator
    already knows attacks happen). Only the localization channel carries
    station-identifying information, so MIA should target loc loss alone.
    This also aligns with what the dual-channel unlearning actually protects.

    Returns dict with mia_forget, mia_retain, mia_overall (same keys as the
    single-channel variant for backward compatibility with result schema).
    """
    model.eval()
    crit = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights)
    all_losses, all_labels = [], []

    with torch.no_grad():
        for loader, lbl in [(member_loader, 1), (non_member_loader, 0)]:
            for batch in loader:
                batch = batch.to(device)
                loc_logits, _ = model(batch)
                loss = crit(loc_logits, batch.y)  # [B, n_labels]
                all_losses.append(loss.cpu())
                all_labels.append(torch.full((loss.size(0),), lbl))

    losses = torch.cat(all_losses).numpy()
    labels = torch.cat(all_labels).numpy()

    if np.any(np.isnan(losses)) or len(np.unique(labels)) < 2:
        return {'mia_forget': 0.5, 'mia_retain': 0.5, 'mia_overall': 0.5}

    def _auc(col_idx):
        if not col_idx:
            return np.nan
        scores = -losses[:, col_idx].mean(axis=1)
        return float(roc_auc_score(labels, scores))

    n_labels = losses.shape[1]
    all_idx = list(range(n_labels))

    if forget_label_idx is not None:
        retain_idx = [i for i in all_idx if i not in forget_label_idx]
        mia_forget = _auc(forget_label_idx)
        mia_retain = _auc(retain_idx)
    else:
        mia_forget = np.nan
        mia_retain = np.nan

    mia_overall = _auc(all_idx)
    return {'mia_forget': mia_forget, 'mia_retain': mia_retain,
            'mia_overall': mia_overall}
