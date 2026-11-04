import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve

def run_epoch(model, loader, criterion_m, criterion_a, device, optimizer=None, scaler=None, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    m_preds, m_labels_all, a_preds, a_labels_all, a_probs = [], [], [], [], []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
        for x, m_y, a_y in pbar:
            x, m_y, a_y = x.to(device, non_blocking=True), m_y.to(device, non_blocking=True), a_y.to(device, non_blocking=True)
            if is_train: optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                m_logits, a_logits = model(x)
                loss = criterion_m(m_logits, m_y) + criterion_a(a_logits, a_y)
            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            m_preds.extend(m_logits.argmax(dim=1).detach().cpu().numpy())
            m_labels_all.extend(m_y.detach().cpu().numpy())
            a_probs.extend(F.softmax(a_logits.float(), dim=1)[:, 1].detach().cpu().numpy())
            a_labels_all.extend(a_y.detach().cpu().numpy())
            if is_train: pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})
    if len(m_labels_all) == 0: return 0.0, 0.0, 0.0, 0.0, 0.0
    m_acc = accuracy_score(m_labels_all, m_preds)
    m_f1 = f1_score(m_labels_all, m_preds, average="macro", zero_division=0)
    a_auc = roc_auc_score(a_labels_all, a_probs) if len(set(a_labels_all)) > 1 else 0.5
    if not is_train and len(set(a_labels_all)) > 1:
        precisions, recalls, thresholds = precision_recall_curve(a_labels_all, a_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        a_preds = (np.array(a_probs) > best_threshold).astype(int)
        a_f1 = max(f1_scores)
        print(f"  [Tuning] 最佳阈值: {best_threshold:.4f} | 最高 A-F1: {a_f1:.4f}")
    else:
        a_preds = (np.array(a_probs) > 0.5).astype(int)
        a_f1 = f1_score(a_labels_all, a_preds, zero_division=0)
    return total_loss / len(loader.dataset), m_acc, m_f1, a_f1, a_auc