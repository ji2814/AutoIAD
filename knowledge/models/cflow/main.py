import json
import einops

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from .torch_model import CflowModel
from .utils import positional_encoding_2d, get_logp
from sklearn.metrics import roc_auc_score, f1_score

from your_dataset_module import Dataset, get_dataloaders  # 你需要提供数据集


def train(model, dataloader, optimizer, device, fiber_batch_size, condition_vector):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch.images.to(device)
        activation = model.encoder(images)

        for layer_idx, layer in enumerate(model.pool_layers):
            encoder_activations = activation[layer].detach()
            batch_size, dim_feature_vector, im_height, im_width = (
                encoder_activations.size()
            )
            embedding_length = batch_size * im_height * im_width

            pos_encoding = einops.repeat(
                positional_encoding_2d(condition_vector, im_height, im_width).unsqueeze(
                    0
                ),
                "b c h w -> (tile b) c h w",
                tile=batch_size,
            ).to(device)

            c_r = einops.rearrange(pos_encoding, "b c h w -> (b h w) c")
            e_r = einops.rearrange(encoder_activations, "b c h w -> (b h w) c")

            perm = torch.randperm(embedding_length)
            decoder = model.decoders[layer_idx]

            fiber_batches = embedding_length // fiber_batch_size
            if fiber_batches <= 0:
                raise ValueError("Fiber batch size too large.")

            for batch_num in range(fiber_batches):
                optimizer.zero_grad()
                if batch_num < fiber_batches - 1:
                    idx = torch.arange(
                        batch_num * fiber_batch_size, (batch_num + 1) * fiber_batch_size
                    )
                else:
                    idx = torch.arange(batch_num * fiber_batch_size, embedding_length)

                c_p = c_r[perm[idx]]
                e_p = e_r[perm[idx]]

                p_u, log_jac_det = decoder(e_p, [c_p])
                decoder_log_prob = get_logp(dim_feature_vector, p_u, log_jac_det)
                log_prob = decoder_log_prob / dim_feature_vector

                loss = -F.logsigmoid(log_prob).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")


def test(model, dataloader, device, threshold=0.5):
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

    print(f"[Test] AUROC: {auroc:.4f}, F1 Score (th={threshold}): {f1:.4f}")
    return auroc, f1


def main():
    # 超参数，可以被优化，优化后的值会被报告给manager
    lr = 1e-4
    batch_size = 4
    fiber_batch_size = 64
    condition_vector = 128
    num_epochs = 20

    # 默认值
    backbone = "wide_resnet50_2"
    layers = ["layer2", "layer3"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = CflowModel(
        backbone=backbone,
        pre_trained=True,
        layers=layers,
        fiber_batch_size=fiber_batch_size,
        decoder="freia-cflow",
        condition_vector=condition_vector,
    ).to(device)

    # 优化器
    decoder_params = []
    for d in model.decoders:
        decoder_params += list(d.parameters())
    optimizer = optim.Adam(decoder_params, lr=lr)

    # 数据加载器
    train_loader, test_loader = get_dataloaders(
        csv_path="path/to/dataset.csv",
        root_dir="path/to/dataset",
        batch_size=batch_size,
    )

    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train(
            model, train_loader, optimizer, device, fiber_batch_size, condition_vector
        )

    # 保存模型
    torch.save(model.state_dict(), "workspace/model.pth")

    # 测试
    f1, auroc = test(model, test_loader, device)

    # 保存结果
    results = {"F1 Score": f1, "AUROC": auroc}
    results["model"] = "cflow"
    results["hyperparameters"] = {
        "lr": lr,
        "batch_size": batch_size,
        "fiber_batch_size": fiber_batch_size,
        "condition_vector": condition_vector,
        "num_epochs": num_epochs,
    }
    with open("workspace/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
