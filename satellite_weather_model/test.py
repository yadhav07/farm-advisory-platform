"""Satellite / Weather Imagery Model - Inference & Evaluation.

Usage examples:
    python test.py                          # evaluate the test set
    python test.py --single path/to/img.jpg # classify a single image
    python test.py --single path --save-cm outputs/confusion_matrix.csv
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from main import Net, EVAL_TRANSFORM, BATCH_SIZE, DEVICE, IMG_SIZE
except ImportError:
    raise

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'satellite_weather_model.pth')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')


def load_model(device=DEVICE):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f'Checkpoint not found at {CHECKPOINT_PATH}. Run main.py to train first.'
        )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    class_names = checkpoint['class_names']
    model = Net(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names


def infer_image(image_path, device=DEVICE):
    model, class_names = load_model(device)
    image = Image.open(image_path).convert('RGB')
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    top_idx = int(np.argmax(probs))
    return {
        'image': os.path.basename(image_path),
        'prediction': class_names[top_idx],
        'probability': float(probs[top_idx]),
        'all_probabilities': {cls: float(prob) for cls, prob in zip(class_names, probs)},
    }


def eval_test_set(device=DEVICE, save_cm=None):
    model, class_names = load_model(device)
    test_dir = os.path.join(DATASET_DIR, 'test')
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f'Test directory not found: {test_dir}')
    ds = datasets.ImageFolder(test_dir, transform=EVAL_TRANSFORM)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    print(f'\nTest accuracy : {acc*100:.2f}%')
    print(f'Test precision: {precision*100:.2f}%')
    print(f'Test recall   : {recall*100:.2f}%')
    print(f'Test F1 macro : {f1*100:.2f}%\n')
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    if save_cm:
        os.makedirs(os.path.join(BASE_DIR, 'outputs'), exist_ok=True)
        np.savetxt(save_cm, cm, fmt='%d', delimiter=',')
        print(f'Confusion matrix CSV saved to {save_cm}')
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.title('Weather State Classification - Confusion Matrix')
            plt.xlabel('Predicted'); plt.ylabel('Actual')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            png_path = os.path.splitext(save_cm)[0] + '.png'
            plt.savefig(png_path, dpi=150)
            print(f'Confusion matrix plot saved to {png_path}')
        except Exception as exc:
            print(f'Could not render plot: {exc}')


def main():
    parser = argparse.ArgumentParser(description='Weather state classifier inference')
    parser.add_argument('--single', help='Path to a single image to classify')
    parser.add_argument('--device', default=None, choices=['cpu', 'cuda'])
    parser.add_argument('--save-cm', default=os.path.join(OUTPUTS_DIR, 'confusion_matrix.csv'),
                        help='Path for confusion matrix CSV (test-set eval)')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE

    if args.single:
        result = infer_image(args.single, device=device)
        print(json.dumps(result, indent=2))
    else:
        eval_test_set(device=device, save_cm=args.save_cm)


if __name__ == '__main__':
    main()