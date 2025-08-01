from tqdm import tqdm
import json
import numpy as np

import torch

from .torch_model import PatchcoreModel
from sklearn.metrics import roc_auc_score, f1_score

from your_dataset_module import Dataset, get_dataloaders  # 你需要提供数据集


@torch.no_grad()
def extract_features(model, dataloader, device):
    model.eval()
    embeddings = []

    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch["image"].to(device)
        features = model(images)
        embeddings.append(features.cpu())

    return torch.vstack(embeddings)


@torch.no_grad()
def test(model, dataloader, device):
    model.eval()
    gt_labels = []
    gt_masks = []

    pred_scores = []
    pre_labels = []
    pre_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch.images.to(device)
            gt_labels.extend(batch.label.to(device))
            gt_masks.extend(batch.mask.to(device))

            # Get predictions from model
            preds = model(images)  # Shape depends on your implementation

            pred_scores.extend(preds.pred_score.cpu().numpy())
            pre_labels.extend(preds.pred_label.cpu().numpy())
            pre_masks.extend(preds.pred_mask.cpu().numpy())

    # Metrics
    auroc = roc_auc_score(
        gt_labels, pred_scores
    )  # or roc_auc_score(gt_masks.reshape(len(gt_masks), -1), pre_masks.reshape(len(pre_masks), -1)) for segmentation task
    f1 = f1_score(
        gt_labels, pre_labels
    )  # or f1_score(gt_masks.reshape(len(gt_masks), -1), pre_masks.reshape(len(pre_masks), -1)) for segmentation task

    print(f"[Test] AUROC: {auroc:.4f}, F1 Score (th={0.5}): {f1:.4f}")
    return auroc, f1


def main():
    # 超参数，可以被优化，优化后的值会被报告给manager
    coreset_sampling_ratio = 0.1
    num_neighbors = 9
    batch_size = 8

    # 默认值
    backbone = "wide_resnet50_2"
    pre_trained = True
    layers = ["layer2", "layer3"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型
    model = PatchcoreModel(
        backbone=backbone,
        pre_trained=pre_trained,
        layers=layers,
        num_neighbors=num_neighbors,
    ).to(device)

    # 数据加载器
    train_loader, test_loader = get_dataloaders(
        csv_path="path/to/dataset.csv",
        root_dir="path/to/dataset",
        batch_size=batch_size,
    )

    # Step 1: 特征提取
    print("\nStep 1: Extracting training embeddings...")
    embeddings = extract_features(model, train_loader, device)

    # Step 2: Coreset 采样
    print("Step 2: Applying coreset sampling...")
    model.subsample_embedding(embeddings, coreset_sampling_ratio)

    # Step 3: 测试
    print("Step 3: Running evaluation on test set...")
    auroc, f1 = test(model, test_loader, device)
    
    # 保存结果
    results = {"F1 Score": f1, "AUROC": auroc}
    results["model"] = "patchcore"
    results["hyperparameters"] = {
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
        "batch_size": batch_size,
    }
    with open("workspace/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
