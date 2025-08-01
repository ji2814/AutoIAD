import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from .loss import GeneratorLoss, DiscriminatorLoss

from your_model_path.torch_model import GanomalyModel
from your_dataset_path import Dataset, get_dataloaders  # 你需要提供


def train(model, dataloader, g_loss_fn, d_loss_fn, g_opt, d_opt, device):
    model.train()
    total_g_loss, total_d_loss = 0.0, 0.0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)

        # Forward pass
        padded, fake, latent_i, latent_o = model(images)
        pred_real, _ = model.discriminator(padded)

        # Generator update
        pred_fake, _ = model.discriminator(fake)
        g_loss = g_loss_fn(latent_i, latent_o, padded, fake, pred_real, pred_fake)
        g_opt.zero_grad()
        g_loss.backward(retain_graph=True)
        g_opt.step()

        # Discriminator update
        pred_fake_detach, _ = model.discriminator(fake.detach())
        d_loss = d_loss_fn(pred_real, pred_fake_detach)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()

    avg_g_loss = total_g_loss / len(dataloader)
    avg_d_loss = total_d_loss / len(dataloader)
    print(
        f"[Train] Generator Loss: {avg_g_loss:.4f}, Discriminator Loss: {avg_d_loss:.4f}"
    )


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

            preds = model(images)

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

    print(f"[Test] AUROC: {auroc:.4f}, F1: {f1:.4f}")
    return auroc, f1


def main():
    # 超参数，可以被优化，优化后的值会被报告给manager
    batch_size = 32
    latent_vec_size = 100
    n_features = 64
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    num_epochs = 20

    # 默认值
    input_size = (256, 256)  # 假设输入图像尺寸
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型与损失
    model = GanomalyModel(
        input_size=input_size,
        num_input_channels=3,
        n_features=n_features,
        latent_vec_size=latent_vec_size,
        extra_layers=0,
        add_final_conv_layer=True,
    ).to(device)

    generator_loss = GeneratorLoss(wadv=1, wcon=50, wenc=1)
    discriminator_loss = DiscriminatorLoss()

    g_opt = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, beta2))
    d_opt = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # 数据加载器
    train_loader, test_loader = get_dataloaders(
        csv_path="path/to/dataset.csv",
        root_dir="path/to/dataset",
        batch_size=batch_size,
    )

    # 训练
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train(
            model,
            train_loader,
            generator_loss,
            discriminator_loss,
            g_opt,
            d_opt,
            device,
        )

    # 测试
    auroc, f1 = test(model, test_loader, device)

    # 保存结果
    results = {"F1 Score": f1, "AUROC": auroc}
    results["model"] = "ganomaly"
    results["hyperparameters"] = {
        "batch_size": batch_size,
        "latent_vec_size": latent_vec_size,
        "n_features": n_features,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "num_epochs": num_epochs,
    }
    with open("workspace/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
