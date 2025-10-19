import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricGenerator(nn.Module):
    """
    Generates a user-specific Mahalanobis matrix (W) based on support set embeddings.

    This module takes a set of support embeddings (genuine samples from a user),
    computes a prototype representation (e.g., mean or weighted mean), and then
    passes this prototype through a generator network to produce the parameters
    of the Mahalanobis matrix W. The matrix W defines a custom distance metric
    tailored to the specific user's signature style.
    """
    def __init__(self, embedding_dim, hidden_factor=2, dropout_prob=0.3):
        """
        Initializes the MetricGenerator.

        Args:
            embedding_dim (int): The dimensionality of the input embeddings.
            hidden_factor (int): Factor to determine the size of hidden layers
                                 relative to embedding_dim. Defaults to 2.
            dropout_prob (float): Dropout probability for regularization. Defaults to 0.3.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        hidden_dim = embedding_dim * hidden_factor

        # Small network to generate the Mahalanobis matrix parameters
        # Input: Prototype vector (embedding_dim)
        # Output: Flattened W matrix (embedding_dim * embedding_dim)
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim), # Added another hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * embedding_dim) # Output flattened matrix
        )

        # Optional: Attention mechanism (can be added if needed, kept simple for now)
        # self.attn = nn.Linear(embedding_dim, 1)

    # def forward(self, support_embeddings):
    #     """
    #     Generates the Mahalanobis matrix W for a given support set.

    #     Args:
    #         support_embeddings (torch.Tensor): Embeddings of the support set samples.
    #                                             Shape: (k_shot, embedding_dim).

    #     Returns:
    #         torch.Tensor: The generated Mahalanobis matrix W, ensured to be
    #                       symmetric and positive semi-definite (PSD).
    #                       Shape: (embedding_dim, embedding_dim).
    #     """
    #     # Ensure support_embeddings is not empty
    #     if support_embeddings.shape[0] == 0:
    #         # Handle empty support set: return identity matrix or raise error
    #         print("Warning: MetricGenerator received empty support_embeddings. Returning Identity matrix.")
    #         # Return identity matrix on the same device
    #         return torch.eye(self.embedding_dim, device=support_embeddings.device)

    #     # --- Compute Prototype ---
    #     # Simple mean prototype (most common)
    #     prototype = torch.mean(support_embeddings, dim=0, keepdim=True) # Shape: (1, embedding_dim)

    #     # --- Generate Flattened Matrix ---
    #     # Pass prototype through the generator network
    #     try:
    #          flat_matrix = self.generator(prototype) # Shape: (1, embedding_dim * embedding_dim)
    #     except Exception as e:
    #          print(f"Error during generator forward pass: {e}")
    #          # Fallback to Identity if generator fails
    #          return torch.eye(self.embedding_dim, device=support_embeddings.device)


    #     # Reshape the output into a square matrix
    #     W_raw = flat_matrix.view(self.embedding_dim, self.embedding_dim)

    #     # --- Ensure W is Symmetric and Positive Semi-Definite (PSD) ---
    #     # 1. Ensure Symmetry: W = (W + W^T) / 2
    #     W_sym = (W_raw + W_raw.t()) / 2

    #     # 2. Ensure Positive Semi-Definite (numerically stable approach):
    #     # Add a small multiple of the identity matrix (epsilon * I)
    #     # This slightly shifts eigenvalues to be positive.
    #     epsilon = 1e-6
    #     W_psd = W_sym + (epsilon * torch.eye(self.embedding_dim, device=W_sym.device))

    #     # Alternative PSD method (more complex): Eigenvalue decomposition
    #     # try:
    #     #     L, V = torch.linalg.eigh(W_sym) # Eigenvalue decomposition
    #     #     L_pos = torch.clamp(L, min=epsilon) # Ensure eigenvalues are >= epsilon
    #     #     W_psd = V @ torch.diag(L_pos) @ V.t() # Reconstruct PSD matrix
    #     # except torch.linalg.LinAlgError:
    #     #      print("Warning: Eigenvalue decomposition failed. Falling back to epsilon * I method.")
    #     #      W_psd = W_sym + (epsilon * torch.eye(self.embedding_dim, device=W_sym.device))


    #     return W_psd

    def forward(self, support_embeddings):
        if support_embeddings.shape[0] == 0:
            print("DEBUG (MetricGenerator): Empty support set, returning Identity.")
            return torch.eye(self.embedding_dim, device=support_embeddings.device)

        # --- 1. Compute Weighted Prototype via Attention ---
        attn_scores = self.attn(support_embeddings)
        attn_weights = F.softmax(attn_scores, dim=0)
        prototype = torch.sum(support_embeddings * attn_weights, dim=0, keepdim=True)
        # === DEBUG PRINT 1 ===
        print(f"DEBUG (MetricGenerator): Prototype shape: {prototype.shape}") # Should be [1, 512]

        # --- 2. Generate Matrix ---
        flat_matrix = self.generator(prototype)
        # === DEBUG PRINT 2 ===
        print(f"DEBUG (MetricGenerator): flat_matrix shape (output of self.generator): {flat_matrix.shape}") # Should be [1, 512*512 = 262144]

        # Reshape to a square matrix
        try:
            # === DEBUG PRINT 3 ===
            print(f"DEBUG (MetricGenerator): Reshaping flat_matrix ({flat_matrix.shape}) to ({self.embedding_dim}, {self.embedding_dim})")
            W_raw = flat_matrix.view(self.embedding_dim, self.embedding_dim)
            # === DEBUG PRINT 4 ===
            print(f"DEBUG (MetricGenerator): W_raw shape after view: {W_raw.shape}") # Should be [512, 512]
        except Exception as e:
            print(f"DEBUG (MetricGenerator): ERROR during flat_matrix.view: {e}")
            # Fallback if reshape fails
            return torch.eye(self.embedding_dim, device=support_embeddings.device)


        # --- 3. Ensure W is Symmetric and Positive Semi-Definite (PSD) ---
        W_sym = (W_raw + W_raw.t()) / 2
        epsilon = 1e-6
        W_psd = W_sym + (epsilon * torch.eye(self.embedding_dim, device=W_sym.device))
        # === DEBUG PRINT 5 ===
        print(f"DEBUG (MetricGenerator): Final W_psd shape: {W_psd.shape}") # Should be [512, 512]

        return W_psd