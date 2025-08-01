
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AnomalyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data_frame.loc[idx, 'image_path'])
        image = Image.open(img_name).convert('RGB')

        gt_mask_path = self.data_frame.loc[idx, 'gt_mask']
        gt_mask = Image.new('L', image.size, 0) # Default to a black mask (no anomaly)
        if gt_mask_path != 'none':
            mask_path = os.path.join(self.root_dir, gt_mask_path)
            if os.path.exists(mask_path):
                gt_mask = Image.open(mask_path).convert('L')
            else:
                print(f"Warning: Mask file not found at {mask_path}")

        is_normal = self.data_frame.loc[idx, 'is_normal']
        
        sample = {'image': image, 'gt_mask': gt_mask, 'is_normal': is_normal}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['gt_mask'] = self.transform(sample['gt_mask']) # Apply same transform to mask

        return sample

if __name__ == '__main__':
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

    # Dataset and DataLoader for testing
    csv_file_path = 'workspace/dataset.csv'
    dataset_root_dir = 'D:/aotuIAD/dataset/MVTecAD/bottle' # This assumes the dataset is here
    
    # Check if dataset.csv exists
    if not os.path.exists(csv_file_path):
        print(f"Error: {csv_file_path} not found. Please run the data processing step first to generate it.")
    else:
        print(f"Loading dataset from {csv_file_path} with root directory {dataset_root_dir}")
        anomaly_dataset = AnomalyDataset(
            csv_file=csv_file_path,
            root_dir=dataset_root_dir,
            transform=CombinedTransform(data_transform, mask_transform)
        )

        dataloader = DataLoader(anomaly_dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility

        print(f"Number of samples in dataset: {len(anomaly_dataset)}")
        print(f"Number of batches in dataloader: {len(dataloader)}")

        # Test fetching a batch
        try:
            for i, sample_batched in enumerate(dataloader):
                print(f"Batch {i}:")
                print(f"  Image batch shape: {sample_batched['image'].shape}")
                print(f"  Mask batch shape: {sample_batched['gt_mask'].shape}")
                print(f"  Is normal batch: {sample_batched['is_normal']}")
                if i == 1: # Print first two batches
                    break
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
        print("Dataloader test complete.")
