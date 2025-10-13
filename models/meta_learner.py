import torch
import torch.nn as nn

class MetricGenerator(nn.Module):
    """
    Generate Mahalanobis matrix W from support set embeddings.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim * embedding_dim)
        )
        self.embedding_dim = embedding_dim

    def forward(self, support_embeddings):
        # Calculate prototype (average vector) for user
        prototype = torch.mean(support_embeddings, dim=0, keepdim=True)

        # Generate a flat matrix W
        flat_matrix = self.generator(prototype)

        # Reshape to a square matrix
        # The matrix W needs to be symmetric and positive definite, but for simplicity,
        # we can start with a square matrix and add the following constraints.
        W = flat_matrix.view(self.embedding_dim, self.embedding_dim)

        # Ensure W is symmetric to be a valid metric
        W = (W + W.T) / 2

        return W