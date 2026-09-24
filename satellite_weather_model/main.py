"""Satellite / Weather Imagery Model - Training Pipeline.

Trains a compact CNN (``Net``) to classify weather states from optical
imagery. A real image dataset (cloud / rain / shine / sunrise) is expected
in ``dataset/train|valid|test``. If those folders are empty the script falls
back to an explicit synthetic dataset so the pipeline always runs end-to-end.

Checkpoint:       satellite_weather_model.pth
Figures/logs:     outputs/
"""

import os
import sys
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except Exception:  # pragma: no cover
    HAS_SKLEARN = False

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'satellite_weather_model.pth')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
PATIENCE = 6
SEED = 42

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomCrop((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    NORMALIZE,
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE,
])


class SyntheticWeatherDataset(Dataset):
    """Deterministic procedural imagery used only when no real data exists."""

    def __init__(self, num_samples=200, image_size=(IMG_SIZE, IMG_SIZE), transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        h, w = self.image_size
        # Correlated spatial gradient (simulated cloud / brightness banding).
        base = rng.uniform(0.0, 1.0)
        xx, yy = np.meshgrid(np.linspace(0, rng.uniform(0.3, 1.5), w),
                             np.linspace(0, rng.uniform(0.3, 1.5), h))
        field = np.clip(base + 0.4 * np.sin(1.5 * xx + rng.uniform(0, 6)) *
                        np.cos(1.5 * yy), 0.0, 1.0)
        img = (field[..., None] * np.ones(3) * 255.0).astype(np.uint8)
        image = Image.fromarray(img, mode='RGB')
        if self.transform:
            image = self.transform(image)
        label = int(rng.integers(0, 4))          # cloud / rain / shine / sunrise
        return image, label


class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        feature_h = IMG_SIZE // 4
        feature_w = IMG_SIZE // 4
        self.fc1 = nn.Linear(32 * feature_h * feature_w, 120)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def build_dataloaders():
    """Load real image datasets, or fall back to the synthetic dataset."""
    train_dir = os.path.join(DATASET_DIR, 'train')
    valid_dir = os.path.join(DATASET_DIR, 'valid')
    test_dir = os.path.join(DATASET_DIR, 'test')

    def _dir_has_images(path):
        if not os.path.isdir(path):
            return False
        for sub in os.listdir(path):
            sub_path = os.path.join(path, sub)
            if os.path.isdir(sub_path) and any(
                    f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    for f in os.listdir(sub_path)):
                return True
        return False

    if not all(_dir_has_images(p) for p in (train_dir, valid_dir, test_dir)):
        print('Real dataset folders missing/empty - using synthetic fallback.')
        full = SyntheticWeatherDataset(num_samples=300, transform=TRAIN_TRANSFORM)
        train_ds, valid_ds, test_ds = random_split(
            full, [180, 60, 60],
            generator=torch.Generator().manual_seed(SEED)
        )
        return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
                DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
                DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
                ['cloud', 'rain', 'shine', 'sunrise'])

    train_ds = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    valid_ds = datasets.ImageFolder(valid_dir, transform=EVAL_TRANSFORM)
    test_ds = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORM)
    class_names = train_ds.classes
    print(f'Loaded real dataset - train {len(train_ds)}, valid {len(valid_ds)}, '
          f'test {len(test_ds)} | classes: {class_names}', flush=True)
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
            DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            class_names)


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return (running_loss / total, correct / total, np.array(all_labels), np.array(all_preds))


def save_learning_curves(history):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.title('Loss curves')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train acc')
    plt.plot(history['val_acc'], label='Val acc')
    plt.title('Accuracy curves')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved learning curves to {path}', flush=True)


def save_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Weather State Classification - Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'Saved confusion matrix to {path}', flush=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print(f'Device: {DEVICE}', flush=True)
    train_loader, valid_loader, test_loader, class_names = build_dataloaders()
    num_classes = len(class_names)

    model = Net(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc, best_epoch, patience = 0.0, 0, 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'Epoch [{epoch:02d}/{EPOCHS}] train_acc={train_acc*100:.2f}% '
              f'val_acc={val_acc*100:.2f}% lr={scheduler.get_last_lr()[0]:.2e}', flush=True)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch, patience = val_acc, epoch, 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'architecture': 'Net',
            }, CHECKPOINT_PATH)
            print(f'  -> best checkpoint saved (val acc {val_acc*100:.2f}%)', flush=True)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f'Early stopping at epoch {epoch}', flush=True)
                break

    print(f'Training finished in {(time.time()-start)/60:.2f} minutes.', flush=True)

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)['model_state_dict'])
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
    print(f'\nFinal test accuracy: {test_acc*100:.2f}% (best epoch {best_epoch})', flush=True)

    if HAS_SKLEARN:
        print('\nClassification report:')
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4), flush=True)
        save_confusion_matrix(y_true, y_pred, class_names)
    save_learning_curves(history)
    print('Checkpoint saved at', CHECKPOINT_PATH, flush=True)


if __name__ == '__main__':
    main()