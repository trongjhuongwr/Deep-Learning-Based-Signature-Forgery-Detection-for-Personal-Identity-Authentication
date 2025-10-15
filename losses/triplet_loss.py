from torch.nn import functional as F
import torch

class DistanceNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(DistanceNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.model(x).squeeze(1) 

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0, mode='euclidean', input_dim = None, hidden_dim=256):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()

        if self.mode == 'learnable':
            if input_dim is None:
                raise ValueError("input_dim must be specified for learnable mode")
            self.distance_net = DistanceNet(input_dim, hidden_dim)

    def forward(self, anchor, positive, negative):
        if self.mode == 'euclidean':
            return self.euclidean_loss(anchor, positive, negative)
        elif self.mode == 'cosine':
            return self.cosine_loss(anchor, positive, negative)
        elif self.mode == 'manhattan':
            return self.manhattan_loss(anchor, positive, negative)
        elif self.mode == 'learnable':
            return self.learnable_loss(anchor, positive, negative)
        else:
            raise ValueError("Unsupported mode")

    def euclidean_loss(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)
    
    def cosine_loss(self, anchor, positive, negative):
        # Normalize vectors
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Cosine distance = 1 - cosine similarity
        #cos_distance(x, y) = 1 − x * y / ∥x∥ * ∥y∥
        dist_ap = 1 - torch.sum(anchor * positive, dim=1)
        dist_an = 1 - torch.sum(anchor * negative, dim=1)

        losses = F.relu(dist_ap - dist_an + self.margin)
        return torch.mean(losses)
    
    def manhattan_loss(self, anchor, positive, negative):
        #L1(x,y)=∑ ∣xi −yi∣
        dist_ap = torch.sum(torch.abs(anchor - positive), dim=1)
        dist_an = torch.sum(torch.abs(anchor - negative), dim=1)
        losses = F.relu(dist_ap - dist_an + self.margin)
        return torch.mean(losses)
    
    def learnable_loss(self, anchor, positive, negative):
        d_ap = self.distance_net(anchor, positive)  # distance anchor-positive
        d_an = self.distance_net(anchor, negative)  # distance anchor-negative

        losses = F.relu(d_ap - d_an + self.margin)
        return torch.mean(losses)


def pairwise_mahalanobis_distance(x1, x2, W):
    """
    Tính ma trận khoảng cách Mahalanobis giữa hai tập tensor (embeddings).
    x1: Tensor kích thước (N, D) - tập các anchor
    x2: Tensor kích thước (M, D) - tập các positive hoặc negative
    W: Ma trận Mahalanobis kích thước (D, D)
    Returns: Ma trận khoảng cách kích thước (N, M)
    """
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)  # Tạo ma trận chênh lệch, kích thước: (N, M, D)
    
    # Sử dụng einsum để tính (x1-x2)^T * W * (x1-x2) cho tất cả các cặp
    # 'ijk,kl,ijl->ij' có nghĩa là: nhân ma trận (N, M, D) với (D, D) và với (N, M, D) để ra (N, M)
    dist_mat = torch.einsum('ijk,kl,ijl->ij', diff, W, diff)
    
    return dist_mat

def adaptive_mahalanobis_triplet_loss(anchor_feat, positive_feat, negative_feat, W, margin=0.5):
    # Anchor-Positive Distance
    dist_ap = torch.diag(pairwise_mahalanobis_distance(anchor_feat, positive_feat, W))
    
    # Anchor-Negative Distance
    dist_an = torch.diag(pairwise_mahalanobis_distance(anchor_feat, negative_feat, W))

    # Calculate Triplet Loss
    losses = F.relu(dist_ap - dist_an + margin)

    # Add a small loss so W doesn't become a zero matrix (regularization)
    matrix_reg_loss = torch.norm(W, p=1) * 1e-4

    return torch.mean(losses) + matrix_reg_loss
    

    