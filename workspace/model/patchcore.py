
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .backbone import WiderResNet
from .common import euclidean_dist

class PatchCore(torch.nn.Module):
    def __init__(self, device='cuda', f_coreset=.1, f_reweighting=.3, 
                 backbone_name='wide_resnet50_2', 
                 input_size=(256, 256)):
        super(PatchCore, self).__init__()
        self.device = device
        self.input_size = input_size

        self.backbone = WiderResNet(pretrained=True).to(self.device)
        self.backbone.eval() # Set to eval mode for feature extraction

        self.f_coreset = f_coreset
        self.f_reweighting = f_reweighting

        self.embedding_coreset = None

        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def embed(self, image):
        with torch.no_grad():
            # Expected input for backbone is a batch of tensors
            if len(image.shape) == 3: # Single image, add batch dimension
                image = image.unsqueeze(0)
            
            # Move image to device and normalize if not already
            if image.max() > 1.0 or image.min() < 0.0: # Assume PIL image or unnormalized tensor
                image = self.norm_transform(transforms.ToPILImage()(image.squeeze(0).cpu())).unsqueeze(0).to(self.device)
            else:
                image = image.to(self.device)

            features = self.backbone(image)

            # Extract patches (simple pooling for now)
            # Dilated Average Pooling (from PatchCore official implementation)
            features = F.avg_pool2d(features, 3, 1, 1)

            # Reshape features to (batch_size, num_patches, feature_dim)
            features = features.permute(0, 2, 3, 1).view(features.shape[0], -1, features.shape[1])

            if features.shape[0] > 1: # Batch processing
                embeddings_list = []
                for i in range(features.shape[0]):
                    embeddings_list.append(features[i].reshape(-1, features.shape[2]))
                return torch.cat(embeddings_list,dim=0)
            return features.reshape(-1, features.shape[2])

    def fit(self, training_data_loader):
        self.backbone.eval()
        print("Extracting features from training data...")
        all_embeddings = []
        for i, sample in enumerate(tqdm(training_data_loader)):
            if i >= 10: # Limit for demonstration purposes within code interpreter
                break
            image = sample['image']
            embeddings = self.embed(image)
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Total embeddings extracted: {all_embeddings.shape}")

        # Coreset subsampling (SPADE algorithm in PatchCore paper)
        print("Performing coreset subsampling...")
        # Simple random subsampling for now; replace with k-NN based coreset selection for actual PatchCore
        num_coreset_samples = int(all_embeddings.shape[0] * self.f_coreset)
        
        # If f_coreset > 1.0, it means we don't apply coreset subsampling, and instead we use
        # f_coreset as the number of samples to take randomly from the dataset.
        # This is not how PatchCore does it, but implemented this way to avoid the need to install cv2 for the kNN. 
        # A proper implementation would use a greedy selection based on k-NN distances
        self.embedding_coreset = all_embeddings[np.random.choice(all_embeddings.shape[0], num_coreset_samples, replace=False)]
        self.embedding_coreset = self.embedding_coreset.to(self.device)
        print(f"Coreset size: {self.embedding_coreset.shape}")

    def predict(self, test_data_loader):
        self.backbone.eval()
        print("Predicting on test data...")
        anomaly_scores = []
        gt_labels = []

        for i, sample in enumerate(tqdm(test_data_loader)):
            if i >= 5: # Limit for demonstration purposes
                break
            image = sample['image']
            is_normal = sample['is_normal']
            
            embeddings = self.embed(image)

            # Nearest Neighbor search
            # For simplicity, using brute force L2 dist, replace with FAISS or similar for speed
            dist = euclidean_dist(embeddings, self.embedding_coreset) # (num_patches, num_coreset)
            min_dist = torch.min(dist, dim=1)[0] # (num_patches,)
            
            # Aggregate patch scores into image score (max pooling over patches)
            score = torch.max(min_dist).item()
            
            anomaly_scores.append(score)
            gt_labels.append(is_normal.cpu().numpy())

        return np.array(anomaly_scores), np.concatenate(gt_labels)
