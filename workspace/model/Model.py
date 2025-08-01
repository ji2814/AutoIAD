
import torch
from .patchcore import PatchCore

class Model(torch.nn.Module):
    def __init__(self, device='cuda', f_coreset=.1, f_reweighting=.3,
                 backbone_name='wide_resnet50_2', input_size=(256, 256)):
        super(Model, self).__init__()
        self.patchcore = PatchCore(
            device=device,
            f_coreset=f_coreset,
            f_reweighting=f_reweighting,
            backbone_name=backbone_name,
            input_size=input_size
        )

    def forward(self, x):
        # For anomaly detection, forward might not be used directly
        # Instead, we will use fit and predict methods of PatchCore
        raise NotImplementedError("Forward pass is not implemented for this model. Use .fit() and .predict() methods.")

    def fit(self, training_data_loader):
        self.patchcore.fit(training_data_loader)

    def predict(self, test_data_loader):
        return self.patchcore.predict(test_data_loader)


if __name__ == '__main__':
    # This is a test block to validate the Model class and its integration
    # It assumes Dataloader.py and dataset.csv are available and correct
    import sys
    sys.path.append('./workspace') # Add workspace to sys.path to import Dataloader
    from Dataloader import AnomalyDataset #, CombinedTransform, data_transform, mask_transform
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import os

    print("Running Model.py test block...")

    # Define transformations
    data_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
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


    csv_file_path = 'workspace/dataset.csv'
    dataset_root_dir = 'D:/aotuIAD/dataset/MVTecAD/bottle'

    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found. Please run the data processing step first to generate it.")
    else:
        try:
            anomaly_dataset = AnomalyDataset(
                csv_file=csv_file_path,
                root_dir=dataset_root_dir,
                transform=CombinedTransform(data_transform, mask_transform)
            )

            # Split dataset into train and test loaders (simplified for this test)
            # In a real scenario, you'd have separate train/test csvs or more robust splitting
            train_dataset = torch.utils.data.Subset(anomaly_dataset, range(int(len(anomaly_dataset)*0.8)))
            test_dataset = torch.utils.data.Subset(anomaly_dataset, range(int(len(anomaly_dataset)*0.8), len(anomaly_dataset)))

            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for Windows
            test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

            print("Initializing PatchCore Model...")
            model = Model(device='cpu') # Use CPU for testing if CUDA not guaranteed

            print("Fitting model to training data...")
            model.fit(train_loader)

            print("Predicting on test data...")
            anomaly_scores, gt_labels = model.predict(test_loader)

            print(f"Anomaly scores shape: {anomaly_scores.shape}")
            print(f"Ground truth labels shape: {gt_labels.shape}")
            print("Model test complete.")

        except Exception as e:
            print(f"An error occurred during model test: {e}")

