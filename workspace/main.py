
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import os
import json
from sklearn.metrics import roc_auc_score
import numpy as np

# Assuming Dataloader.py and Model.py are in the workspace directory or accessible via sys.path
from Dataloader import AnomalyDataset #, CombinedTransform, data_transform, mask_transform
from model.Model import Model # model.Model since Model.py is in workspace/model

# --- Configuration ---
DATASET_CSV_PATH = 'workspace/dataset.csv'
DATASET_ROOT_DIR = 'D:/aotuIAD/dataset/MVTecAD/bottle'
MODEL_CHECKPOINT_PATH = 'workspace/model.pth'
RESULTS_FILE_PATH = 'workspace/results.json'

# Anomaly detection specific hyperparameters
# f_coreset and f_reweighting are parameters internal to PatchCore, set in model.Model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4 # Small batch size for demonstration/memory constraints
INPUT_SIZE = (256, 256)

# --- Transformations ---
data_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
])

class CombinedTransform:
    def __init__(self, image_transform, mask_transform):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, sample):
        if 'image' in sample:
            sample['image'] = self.image_transform(sample['image'])
        if 'gt_mask' in sample:
            sample['gt_mask'] = self.mask_transform(sample['gt_mask'])
        return sample

def main():
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    if not os.path.exists(DATASET_CSV_PATH):
        print(f"Error: {DATASET_CSV_PATH} not found. Please run the data processing step first.")
        return

    full_dataset = AnomalyDataset(
        csv_file=DATASET_CSV_PATH,
        root_dir=DATASET_ROOT_DIR,
        transform=CombinedTransform(data_transform, mask_transform)
    )

    # Separate train and test splits based on 'split' column in dataset.csv
    train_indices = full_dataset.data_frame[full_dataset.data_frame['split'] == 'train'].index.tolist()
    test_indices = full_dataset.data_frame[full_dataset.data_frame['split'] == 'test'].index.tolist()

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")

    # --- Model Initialization ---
    model = Model(device=DEVICE, input_size=INPUT_SIZE)
    model.to(DEVICE)

    # --- Training (Fit) ---
    print("Starting model training (feature extraction and coreset building)...")
    model.fit(train_loader)

    # --- Evaluation (Predict) ---
    print("Starting model evaluation...")
    anomaly_scores, gt_labels = model.predict(test_loader)

    # Ensure gt_labels are of type integer/binary for AUROC
    gt_labels = (gt_labels == 1).astype(int) # Good=1, Anomaly=0. AUROC typically expects 1 for positive class (anomaly)

    if len(np.unique(gt_labels)) < 2:
        print("Warning: Only one class present in ground truth labels. Cannot compute AUROC.")
        auroc = None
    else:
        # For AUROC, we want to predict anomaly (1) for abnormal samples
        # The 'is_normal' column is 1 for normal, 0 for anomaly.
        # anomaly_scores should be higher for anomalous samples.
        # So, if gt_labels is 1 for normal and 0 for anomaly,
        # and anomaly_scores are higher for anomaly, then we need to negate gt_labels or
        # adjust the interpretation. Let's assume AUROC expects positive scores for positive class.
        # Here, anomaly_scores are higher for anomalies (good). So, we should flip the gt_labels.
        # If gt_labels is 1 for normal, 0 for anomaly.
        # We want to predict 1 for anomaly. So we use (1 - gt_labels) as true labels for AUROC.
        auroc = roc_auc_score(1 - gt_labels, anomaly_scores)
        print(f"Computed AUROC: {auroc:.4f}")

    # --- Reporting ---
    results = {
        "dataset": "MVTecAD/bottle",
        "model": "PatchCore",
        "metric": "AUROC",
        "auroc": auroc,
        "f_coreset": model.patchcore.f_coreset,
        "f_reweighting": model.patchcore.f_reweighting,
        "device": DEVICE,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset)
    }

    with open(RESULTS_FILE_PATH, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE_PATH}")

    # --- Model Saving ---
    # PatchCore's "model" is primarily its coreset.
    # To save the "model", we save the embedding_coreset.
    # Check if coreset exists before saving
    if model.patchcore.embedding_coreset is not None:
        torch.save({
            'embedding_coreset': model.patchcore.embedding_coreset.cpu(),
            'f_coreset': model.patchcore.f_coreset,
            'f_reweighting': model.patchcore.f_reweighting,
            'input_size': model.patchcore.input_size,
            # You might want to save backbone state_dict too if you plan finer control
            # 'backbone_state_dict': model.patchcore.backbone.state_dict(),
        }, MODEL_CHECKPOINT_PATH)
        print(f"Model coreset and parameters saved to {MODEL_CHECKPOINT_PATH}")
    else:
        print("No coreset found to save. Model was not fully trained or coreset is empty.")


if __name__ == '__main__':
    main()
