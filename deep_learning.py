from data import ISRUC_ECG_Reader
from deep_learning_preprocess import ECGPreprocessor
# from rpeaks import RRExtractor
# from hrv_features import HRVFeatureExtractor
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 使用无界面后端，适配服务器环境
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score,
    cohen_kappa_score,
)
import random

from deep_learning_model import SleepECGCNNLSTM
import torch.nn.functional as F
class ECGSleepDataset(Dataset):
    def __init__(self, all_epochs, all_labels, window_size=11, fs=200,
                 n_fft=256, hop_length=64, win_length=256,
                 fmin=0.5, fmax=40.0, subject_lengths=None):
        """STFT-based ECG sleep dataset.

        all_epochs: (N_total, 6000) — 所有 epoch 按 subject 依次拼接
        all_labels: (N_total,)      — 对应标签
        window_size: 必须为奇数，如 11
        subject_lengths: list[int] or None
            如果提供，则表示各 subject 对应的 epoch 数量，数据在 all_epochs 中
            按该顺序依次拼接。我们会在每个 subject 内部构造中心索引，
            从而保证滑动窗口不会跨越 subject 边界；
            如果为 None，则退回到原有的全局滑窗行为（可能跨 subject）。
        """

        self.window_size = window_size
        self.half_win = window_size // 2
        self.epochs = all_epochs
        self.labels = all_labels

        # 若给出每个 subject 的长度，则在各自的局部范围内构造合法中心索引，
        # 从而避免窗口跨越 subject 边界；否则保持原有行为。
        if subject_lengths is not None:
            self.valid_indices = []
            offset = 0
            for L in subject_lengths:
                if L <= 2 * self.half_win:
                    # 该 subject 太短，无法形成完整窗口，直接跳过
                    offset += L
                    continue
                start_idx = offset + self.half_win
                end_idx = offset + L - self.half_win
                self.valid_indices.extend(range(start_idx, end_idx))
                offset += L
        else:
            self.valid_indices = list(range(self.half_win, len(self.epochs) - self.half_win))

        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

        # 预先算好频率裁剪的bin范围
        freqs = torch.fft.rfftfreq(n_fft, d=1.0/fs)  # (n_fft/2+1,)
        self.f_low = int(torch.searchsorted(freqs, torch.tensor(fmin), right=False))
        self.f_high = int(torch.searchsorted(freqs, torch.tensor(fmax), right=True))

    def _epoch_to_spec(self, x_1d: torch.Tensor):
        # x_1d: (6000,)
        X = torch.stft(
            x_1d, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window, center=False, return_complex=True
        )  # (freq, time)
        power = (X.real**2 + X.imag**2)
        spec = torch.log(power + 1e-6)

        spec = spec[self.f_low:self.f_high]  # (F, T) 只保留 0.5-40Hz
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)  # 每张谱图标准化
        return spec

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        start = center_idx - self.half_win
        end = center_idx + self.half_win + 1

        x = torch.from_numpy(self.epochs[start:end]).float()  # (W, 6000)
        y = int(self.labels[center_idx])

        specs = torch.stack([self._epoch_to_spec(xi) for xi in x], dim=0)  # (W, F, T)
        return specs, torch.tensor(y, dtype=torch.long)
    
def plot_ecg_epoch(
    raw_epoch=None,
    preprocessed_epoch=None,
    sfreq=200,
    epoch_index=0,
    subject_id="Unknown",
    save_path=None,
):
    """
    Plot one ECG epoch (raw and/or preprocessed) for visual inspection.

    Parameters:
        raw_epoch (np.ndarray, optional): Raw ECG signal of shape (n_samples,)
        preprocessed_epoch (np.ndarray, optional): Preprocessed ECG signal of shape (n_samples,)
        sfreq (int): Sampling frequency in Hz (e.g., 200)
        epoch_index (int): Index of the epoch (for labeling)
        subject_id (str): Subject ID (for labeling)
        save_path (str or Path, optional): If provided, save the figure to this path.
    """
    if raw_epoch is None and preprocessed_epoch is None:
        raise ValueError("At least one of raw_epoch or preprocessed_epoch must be provided.")

    n_samples = len(raw_epoch) if raw_epoch is not None else len(preprocessed_epoch)
    time_axis = np.arange(n_samples) / sfreq  # in seconds

    plt.figure(figsize=(12, 4))
    
    if raw_epoch is not None:
        plt.plot(time_axis, raw_epoch, label='Raw ECG', color='red', linewidth=1.0)
    
    if preprocessed_epoch is not None:
        plt.plot(time_axis, preprocessed_epoch, label='Preprocessed ECG', color='blue', linewidth=1.0)

    plt.title(f"ECG Signal (30s example)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")

    plt.show()

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, n_classes=5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return acc, f1_macro, all_labels, all_preds

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()


def load_latest_model_and_report(model_dir="aHza3eIYheX/models", val_loader=None, device=None, plot_cm=True, target_names=None):
    """Load the most recently saved .pth checkpoint from `model_dir`, rebuild
    the model from the saved state dict, evaluate on `val_loader`, and
    print classification report and confusion matrix.

    Parameters:
        model_dir (str or Path): directory containing .pth checkpoints
        val_loader (DataLoader): validation DataLoader to evaluate on
        device (torch.device or str, optional): device to load model to
        plot_cm (bool): whether to show a plotted confusion matrix
        target_names (list[str], optional): names for classes in report
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        print(f"Model directory not found: {model_dir}")
        return
    pths = list(model_dir.glob("*.pth"))
    if len(pths) == 0:
        print(f"No .pth files found in {model_dir}")
        return
    latest = max(pths, key=lambda p: p.stat().st_mtime)
    print(f"Loading latest checkpoint: {latest}")

    map_loc = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(latest, map_location=map_loc)
    # support both raw state_dict saved and dict with 'model_state_dict'
    state = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

    # try to infer n_classes from classifier weight shape
    n_classes = None
    for k, v in state.items():
        if k.endswith('classifier.weight') or 'classifier.weight' in k:
            try:
                n_classes = int(v.shape[0])
            except Exception:
                n_classes = None
            break
    if n_classes is None:
        n_classes = 5
        print(f"Could not infer n_classes from checkpoint; defaulting to {n_classes}")

    device_t = torch.device(map_loc)
    model = SleepECGCNNLSTM(input_length=6000, n_classes=n_classes).to(device_t)
    try:
        model.load_state_dict(state)
    except Exception as e:
        # attempt to strip possible prefixes like 'module.'
        new_state = {}
        for k, v in state.items():
            new_k = k.replace('module.', '')
            new_state[new_k] = v
        model.load_state_dict(new_state)

    if val_loader is None:
        print("val_loader is required to evaluate the loaded model.")
        return

    acc, f1, y_true, y_pred = evaluate(model, val_loader, device_t, n_classes=n_classes)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    if plot_cm:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        ticks = np.arange(n_classes)
        plt.xticks(ticks, [str(t) for t in ticks], rotation=45)
        plt.yticks(ticks, [str(t) for t in ticks])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

def main():
    # 以当前脚本所在目录作为工程根目录，避免依赖运行时工作目录
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir / "ISRUC/subjects"

    reader = ISRUC_ECG_Reader(root_dir)
    data = reader.load_all()
    preprocessor = ECGPreprocessor(sfreq=200)
    n_classes = 5
    all_subjects = []

    window_size = 11  # must be odd

    # Only ONE pass: preprocess + label mapping + filtering + append
    for sid, subject_data in data.items():
        ecg_epochs = subject_data["ecg_epochs"]
        labels = subject_data["labels"].copy()  # avoid modifying original

        # Map ISRUC label 5 (REM) to 4
        labels[labels == 5] = 4

        # Keep only labels in [0, 4]
        valid_mask = (labels >= 0) & (labels <= 4)

        # Preprocess all epochs
        n_epochs = ecg_epochs.shape[0]
        preprocessed_epochs = np.zeros_like(ecg_epochs, dtype=np.float32)
        for i in range(n_epochs):
            preprocessed_epochs[i] = preprocessor.preprocess_ecg_epoch(ecg_epochs[i])

        # Apply mask to both epochs and labels
        preprocessed_epochs = preprocessed_epochs[valid_mask]
        labels = labels[valid_mask]

        all_subjects.append((sid, preprocessed_epochs, labels))

    print(len(all_subjects), "subjects loaded and preprocessed.")

    all_labels = np.concatenate([labels for _, _, labels in all_subjects])
    print("Label distribution:", np.bincount(all_labels))
    print("Unique labels:", np.unique(all_labels))
    # plot_ecg_epoch(
    #     raw_epoch=all_subjects[0][1][0],
    #     preprocessed_epoch=None,
    #     subject_id=all_subjects[0][0],
    #     epoch_index=0,
    # )
    # Step 2: Subject IDs for 5-fold CV
    subject_ids = [sid for sid, _, _ in all_subjects]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store results across folds
    fold_results = []

    # 保存模型和 PR 曲线的目录
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    pr_dir = base_dir / "pr_curves"
    pr_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
        print(f"\n{'='*20} Fold {fold+1}/5 {'='*20}")

        # Build training and validation data
        train_epochs, train_labels = [], []
        val_epochs, val_labels = [], []

        for i in train_idx:
            _, epochs, labels = all_subjects[i]
            train_epochs.append(epochs)
            train_labels.append(labels)
        for i in val_idx:
            _, epochs, labels = all_subjects[i]
            val_epochs.append(epochs)
            val_labels.append(labels)

        # 记录每个 subject 的 epoch 数量，用于在 Dataset 中禁止窗口跨 subject
        train_subject_lengths = [e.shape[0] for e in train_epochs]
        val_subject_lengths = [e.shape[0] for e in val_epochs]

        # Concatenate across subjects
        train_epochs = np.concatenate(train_epochs, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        val_epochs = np.concatenate(val_epochs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        # Filter out invalid labels (e.g., label = -1 for undefined stages)
        valid_train = train_labels >= 0
        valid_val = val_labels >= 0
        train_epochs, train_labels = train_epochs[valid_train], train_labels[valid_train]
        val_epochs, val_labels = val_epochs[valid_val], val_labels[valid_val]

        print("Train label dist:", np.bincount(train_labels, minlength=n_classes))
        print("Val   label dist:", np.bincount(val_labels, minlength=n_classes))

        # Create datasets，传入各 subject 的长度，确保滑动窗口不会跨 subject
        train_dataset = ECGSleepDataset(train_epochs, train_labels,
                        window_size=window_size,
                        subject_lengths=train_subject_lengths)
        val_dataset = ECGSleepDataset(val_epochs, val_labels,
                          window_size=window_size,
                          subject_lengths=val_subject_lengths)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("⚠️ Skipping fold due to empty dataset.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset,   batch_size=128, shuffle=False, num_workers=4)

        model = SleepECGCNNLSTM(n_classes=n_classes).to(device)

        class_counts = np.bincount(train_labels, minlength=n_classes)
        weights = 1.0 / np.sqrt(class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        weights = np.clip(weights, a_min=0.5, a_max=1.5)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # ✅ optimizer/scheduler 只初始化一次
        optimizer = torch.optim.Adam(model.parameters(), lr=7e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=5,
            factor=0.5,
            threshold=1e-3,
            cooldown=1
        )

        best_f1 = -1.0
        patience_counter = 0
        best_model_path = model_dir / f"fold_{fold+1}_best_f1.pth"

        for epoch in range(40):
            # ✅ 只更新 loss 的权重，不要重建 optimizer
            alpha = min(1.0, (epoch + 1) / 3.0)   # 建议从 epoch1 就开始递增，别让 epoch0=0 太极端
            w = 1.0 + alpha * (weights - 1.0)
            criterion = nn.CrossEntropyLoss(weight=w)

            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

            # 这里最好让 evaluate 返回 val_loss 也行，但先保持不动
            val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

            scheduler.step(val_f1)

            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save({
                    'fold': fold + 1,
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                }, best_model_path)
                print(f"✅ New best F1: {best_f1:.4f}, saved to {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered.")
                    break

            

        # Final evaluation with best model
        model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
        final_acc, final_f1, y_true, y_pred = evaluate(model, val_loader, device)
        # Balanced Accuracy
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # 计算用于绘制 PR 曲线的预测概率
        model.eval()
        all_probs = []
        all_labels_for_pr = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                logits_batch = model(x_batch)
                probs_batch = torch.softmax(logits_batch, dim=1).cpu().numpy()
                all_probs.append(probs_batch)
                all_labels_for_pr.extend(y_batch.numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels_for_pr = np.array(all_labels_for_pr)

        # 绘制每一类的一对多 PR 曲线
        plt.figure(figsize=(8, 6))
        for c in range(n_classes):
            y_true_c = (all_labels_for_pr == c).astype(int)
            # 若该类在验证集中完全不存在，则跳过，避免异常
            if y_true_c.sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(y_true_c, all_probs[:, c])
            ap = average_precision_score(y_true_c, all_probs[:, c])
            plt.plot(recall, precision, lw=2, label=f"Class {c} (AP={ap:.2f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Fold {fold+1} Precision-Recall Curves")
        plt.legend(loc="best")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        pr_curve_path = pr_dir / f"fold_{fold+1}_pr_curve.png"
        plt.savefig(pr_curve_path, dpi=150)
        plt.close()
        print(f"Saved PR curve for fold {fold+1} to {pr_curve_path}")

        fold_results.append({
            'fold': fold + 1,
            'accuracy': final_acc,
            'f1_macro': final_f1,
            'balanced_accuracy': bal_acc,
            'kappa': kappa,
            'y_true': y_true,
            'y_pred': y_pred
        })
        print(f"✅ Fold {fold+1} Final | Acc: {final_acc:.4f}, F1: {final_f1:.4f}, Balanced Acc: {bal_acc:.4f}, Kappa: {kappa:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        print(cm)

    # Aggregate results
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
    avg_bal_acc = np.mean([r['balanced_accuracy'] for r in fold_results])
    avg_kappa = np.mean([r['kappa'] for r in fold_results])
    print(f"\n{'='*50}")
    print(f"Overall 5-Fold CV Results:")
    print(f"Accuracy: {avg_acc:.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"F1-Macro: {avg_f1:.4f} ± {np.std([r['f1_macro'] for r in fold_results]):.4f}")
    print(f"Balanced Accuracy: {avg_bal_acc:.4f} ± {np.std([r['balanced_accuracy'] for r in fold_results]):.4f}")
    print(f"Kappa: {avg_kappa:.4f} ± {np.std([r['kappa'] for r in fold_results]):.4f}")

    # Optional: Print final classification report from last fold or aggregate
    # (For full aggregation, collect all y_true/y_pred across folds)

if __name__ == "__main__":
    main()