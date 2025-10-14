# models/meta_learner.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricGenerator(nn.Module):
    """
    Sinh ra ma trận Mahalanobis W từ support set embeddings sử dụng Attention.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        # Lớp tuyến tính để tính điểm attention
        self.attn = nn.Linear(embedding_dim, 1)

        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, support_embeddings):
        # Tính attention weights
        # (K_SHOT, EMBEDDING_DIM) -> (K_SHOT, 1)
        attn_scores = self.attn(support_embeddings)
        # Áp dụng softmax để trọng số có tổng bằng 1
        attn_weights = F.softmax(attn_scores, dim=0)

        # Tính prototype có trọng số (weighted average)
        # (K_SHOT, EMBEDDING_DIM) * (K_SHOT, 1) -> (K_SHOT, EMBEDDING_DIM)
        # torch.sum(...) -> (1, EMBEDDING_DIM)
        prototype = torch.sum(support_embeddings * attn_weights, dim=0, keepdim=True)

        flat_matrix = self.generator(prototype)
        W = flat_matrix.view(self.embedding_dim, self.embedding_dim)

        # Đảm bảo W đối xứng và xác định dương (quan trọng cho sự ổn định)
        W = (W + W.T) / 2
        W = W + torch.eye(self.embedding_dim, device=W.device) * 1e-5 # Thêm một giá trị nhỏ vào đường chéo

        return W