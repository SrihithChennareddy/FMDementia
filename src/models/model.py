import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from PIL import Image
import numpy as np

# -----------------------------
# Configuration
# -----------------------------
DATA_DIRS = {
    "OASIS-1": "./datasets/OASIS1",
    "OASIS-2": "./datasets/OASIS2"
}
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 2  # HC vs DM
KFOLD_SPLITS = 5

# -----------------------------
# Dataset Class
# -----------------------------
class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for label, cls in enumerate(['HC', 'DM']):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# Model Definition
# -----------------------------
def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

# -----------------------------
# Training & Evaluation Functions
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    preds, targets, probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            prob = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
            pred = torch.argmax(outputs, dim=1).cpu().numpy()
            preds.extend(pred)
            probs.extend(prob)
            targets.extend(labels.numpy())
    accuracy = accuracy_score(targets, preds)*100
    precision = precision_score(targets, preds)*100
    recall = recall_score(targets, preds)*100
    f1 = f1_score(targets, preds)*100
    auc = roc_auc_score(targets, probs)*100
    return accuracy, precision, recall, f1, auc

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# 5-Fold Cross-Validation
# -----------------------------
def cross_val_results(data_dir):
    dataset = MRIDataset(data_dir, transform=transform)
    X = np.arange(len(dataset))
    y = np.array(dataset.labels)

    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)
    accs, precisions, recalls, f1s, aucs = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        model = get_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(EPOCHS):
            train_one_epoch(model, train_loader, criterion, optimizer)

        metrics = evaluate(model, val_loader)
        accs.append(metrics[0])
        precisions.append(metrics[1])
        recalls.append(metrics[2])
        f1s.append(metrics[3])
        aucs.append(metrics[4])

    return np.mean(accs), np.mean(precisions), np.mean(recalls), np.mean(f1s), np.mean(aucs)

# -----------------------------
# Cross-Domain Evaluation
# -----------------------------
def cross_domain(train_dir, test_dir):
    train_dataset = MRIDataset(train_dir, transform=transform)
    test_dataset = MRIDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer)

    return evaluate(model, test_loader)

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Experimental results on OASIS-1 and OASIS-2 datasets using 5-fold cross-validation")
    print("Dataset\tAccuracy\tPrecision\tRecall\tF1-score\tAUC-ROC")

    # 5-Fold CV
    results = {}
    for ds_name, ds_dir in DATA_DIRS.items():
        metrics = cross_val_results(ds_dir)
        results[ds_name] = metrics
        print(f"{ds_name}\t{metrics[0]:.1f}%\t{metrics[1]:.1f}%\t{metrics[2]:.1f}%\t{metrics[3]:.1f}%\t{metrics[4]:.1f}%")

    # Average
    avg_metrics = np.mean(list(results.values()), axis=0)
    print(f"Average\t{avg_metrics[0]:.1f}%\t{avg_metrics[1]:.1f}%\t{avg_metrics[2]:.1f}%\t{avg_metrics[3]:.1f}%\t{avg_metrics[4]:.1f}%")

    # Cross-Domain Evaluation
    print("\nCross-Domain Evaluation")
    print("Train / Test\tAccuracy\tPrecision\tRecall\tF1-score\tAUC-ROC")

    cd_1_2 = cross_domain(DATA_DIRS["OASIS-1"], DATA_DIRS["OASIS-2"])
    cd_2_1 = cross_domain(DATA_DIRS["OASIS-2"], DATA_DIRS["OASIS-1"])

    print(f"OASIS-1 / OASIS-2\t{cd_1_2[0]:.1f}%\t{cd_1_2[1]:.1f}%\t{cd_1_2[2]:.1f}%\t{cd_1_2[3]:.1f}%\t{cd_1_2[4]:.1f}%")
    print(f"OASIS-2 / OASIS-1\t{cd_2_1[0]:.1f}%\t{cd_2_1[1]:.1f}%\t{cd_2_1[2]:.1f}%\t{cd_2_1[3]:.1f}%\t{cd_2_1[4]:.1f}%")

