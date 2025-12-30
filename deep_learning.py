from data import ISRUC_ECG_Reader
from deep_learning_preprocess import ECGPreprocessor
# from rpeaks import RRExtractor
# from hrv_features import HRVFeatureExtractor
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import random

from deep_learning_model import SleepECGCNNLSTM
import torch.nn.functional as F
class ECGSleepDataset(Dataset):
    def __init__(self, all_epochs, all_labels, window_size=5):
        """
        all_epochs: (N_total, 6000) — 所有 epoch 拼接
        all_labels: (N_total,)      — 对应标签
        window_size: 必须为奇数，如 5
        """
        self.window_size = window_size
        self.half_win = window_size // 2
        self.epochs = all_epochs
        self.labels = all_labels
        self.valid_indices = list(range(self.half_win, len(self.epochs) - self.half_win))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        center_idx = self.valid_indices[idx]
        start = center_idx - self.half_win
        end = center_idx + self.half_win + 1
        x = self.epochs[start:end]          # (5, 6000)
        y = self.labels[center_idx]         # scalar
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)
    
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
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
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
    root_dir = Path("big/ISRUC/subjects")

    reader = ISRUC_ECG_Reader(root_dir)
    data = reader.load_all()
    preprocessor = ECGPreprocessor(sfreq=200)

    all_subjects = []

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

        # Create datasets
        train_dataset = ECGSleepDataset(train_epochs, train_labels, window_size=5)
        val_dataset = ECGSleepDataset(val_epochs, val_labels, window_size=5)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("⚠️ Skipping fold due to empty dataset.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

        # Model, loss, optimizer
        # Derive number of classes from training labels to ensure consistency
        n_classes = int(train_labels.max()) + 1
        model = SleepECGCNNLSTM(input_length=6000, n_classes=n_classes).to(device)
        class_counts = np.bincount(train_labels, minlength=n_classes)
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)  # normalize
        weights = torch.FloatTensor(weights).to(device)
        # criterion = nn.CrossEntropyLoss(weight=weights)
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        best_f1 = 0
        best_acc = 0
        best_model_path = f"big/models/fold_{fold+1}_best_acc.pth"
        Path("models").mkdir(exist_ok=True)
        patience_counter = 0
        
        for epoch in range(40):  # max 50 epochs
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

            scheduler.step(val_f1)
            print(f"Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered.")
                    break
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save({
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                    }, best_model_path)
                    print(f"✅ New best Acc: {best_acc:.4f}, saved to {best_model_path}")

        # Final evaluation with best model
        model.load_state_dict(best_model_state)
        final_acc, final_f1, y_true, y_pred = evaluate(model, val_loader, device)
        fold_results.append({
            'fold': fold + 1,
            'accuracy': final_acc,
            'f1_macro': final_f1,
            'y_true': y_true,
            'y_pred': y_pred
        })
        print(f"✅ Fold {fold+1} Final | Acc: {final_acc:.4f}, F1: {final_f1:.4f}")

    # Aggregate results
    avg_acc = np.mean([r['accuracy'] for r in fold_results])
    avg_f1 = np.mean([r['f1_macro'] for r in fold_results])
    print(f"\n{'='*50}")
    print(f"Overall 5-Fold CV Results:")
    print(f"Accuracy: {avg_acc:.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"F1-Macro: {avg_f1:.4f} ± {np.std([r['f1_macro'] for r in fold_results]):.4f}")

    # Optional: Print final classification report from last fold or aggregate
    # (For full aggregation, collect all y_true/y_pred across folds)

if __name__ == "__main__":
    main()