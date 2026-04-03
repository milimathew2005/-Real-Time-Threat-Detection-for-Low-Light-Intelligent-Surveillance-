import os, argparse, numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import SPLITS_DIR, BEST_MODEL_PATH, LOG_DIR, BATCH_SIZE, SEQ_LEN, OVERLAP, CLASSES
from dataset import UCFCrimeDataset, collate_fn
from model import build_model

@torch.no_grad()
def predict(model, loader, device, num_classes=2, threshold=0.5):
    model.eval()
    preds, labels = ([], [])
    for seqs, lbs in loader:
        logits, _ = model(seqs.to(device))
        if num_classes == 2:
            probs = torch.softmax(logits, dim=1)
            batch_preds = (probs[:, 1] >= threshold).int()
            preds.extend(batch_preds.cpu().tolist())
        else:
            preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(lbs.tolist())
    return (np.array(preds), np.array(labels))

def plot_cm(y_true, y_pred, names, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm.astype(float) / cm.sum(1, keepdims=True), annot=cm, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names, ax=ax)
    ax.set(xlabel='Predicted', ylabel='True', title='UCF-Crime LSTM — Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Confusion matrix → {path}')

def evaluate(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    test_csv = args.test_csv or os.path.join(SPLITS_DIR, 'test.csv')
    ds = UCFCrimeDataset(test_csv, SEQ_LEN, OVERLAP, augment=False)
    dl = DataLoader(ds, args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)
    print(f'Test windows: {len(ds):,}')
    if len(ds) == 0:
        raise RuntimeError(f'No test windows found. Check that paths in {test_csv} point to existing .npy files.')
    model = build_model(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f'Checkpoint: epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', 0):.4f}')
    preds, labels = predict(model, dl, device, num_classes=args.num_classes, threshold=args.threshold)
    acc = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average='macro', zero_division=0)
    f1_wt = f1_score(labels, preds, average='weighted', zero_division=0)
    present_labels = sorted(set(labels) | set(preds))
    if args.num_classes == 2:
        name_map = {0: 'Normal', 1: 'Anomaly'}
        present_names = [name_map[i] for i in present_labels]
    else:
        present_names = [CLASSES[i] for i in present_labels]
    report = classification_report(labels, preds, labels=present_labels, target_names=present_names, zero_division=0)
    summary = f'Test Accuracy : {acc * 100:.2f}%\nMacro F1      : {f1_mac * 100:.2f}%\nWeighted F1   : {f1_wt * 100:.2f}%\n\n{report}'
    print('\n' + summary)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, 'test_metrics.txt'), 'w') as f:
        f.write(summary)
    plot_cm(labels, preds, present_names, os.path.join(LOG_DIR, 'confusion_matrix.png'))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default=BEST_MODEL_PATH)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    p.add_argument('--test_csv', default='D:\\Threat Detection\\splits\\test_binary.csv', help='Override default test CSV path')
    p.add_argument('--num_classes', type=int, default=2, help='Override NUM_CLASSES (e.g. 2 for binary)')
    p.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for predicting Anomaly class in binary mode (default: 0.5)')
    evaluate(p.parse_args())