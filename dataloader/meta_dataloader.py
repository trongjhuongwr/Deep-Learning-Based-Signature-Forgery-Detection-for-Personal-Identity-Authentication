import torch
from torch.utils.data import Dataset
import random
import json
from PIL import Image

class SignatureEpisodeDataset(Dataset):
    def __init__(self, split_file_path, split_name, k_shot=5, n_query=5, transform=None):
        """
        Args:
            split_file_path (str): Path to cedar_meta_split.json file.
            split_name (str): 'meta-train' or 'meta-test'.
            k_shot (int): Number of samples in support set (number of real signatures).
            n_query (int): Number of query samples for each type (real/fake).
            transform (callable, optional): Transform applied to the image.
        """
        with open(split_file_path, 'r') as f:
            self.data = json.load(f)[split_name]

        self.user_ids = list(self.data.keys())
        self.k_shot = k_shot
        self.n_query = n_query
        self.transform = transform
        self.length = len(self.user_ids) # In meta-learning, 1 epoch usually goes through all "tasks" (users)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.data[user_id]

        genuine_paths = user_data['genuine']
        forgery_paths = user_data['forgery']

        if len(genuine_paths) < self.k_shot + self.n_query:
            raise ValueError(f"User {user_id} does not have enough genuine samples. Needs {self.k_shot + self.n_query}, has {len(genuine_paths)}.")
        if len(forgery_paths) < self.n_query:
            raise ValueError(f"User {user_id} does not have enough forgery paths. Needs {self.n_query}, has {len(forgery_paths)}.")

        random.shuffle(genuine_paths)
        random.shuffle(forgery_paths)

        # Create Support Set (only includes real signatures)
        support_paths = genuine_paths[:self.k_shot]

        # Create Query Set (including real and fake)
        query_genuine_paths = genuine_paths[self.k_shot : self.k_shot + self.n_query]
        query_forgery_paths = forgery_paths[:self.n_query]
        query_paths = query_genuine_paths + query_forgery_paths

        # Create label: 1 for real, 0 for fake
        support_labels = [1] * len(support_paths)
        query_labels = [1] * len(query_genuine_paths) + [0] * len(query_forgery_paths)

        # Auxiliary function to load and transform images
        def _load_images(paths):
            images = [Image.open(p).convert('L') for p in paths]
            if self.transform:
                images = [self.transform(img) for img in images]
            return torch.stack(images)

        support_images = _load_images(support_paths)
        query_images = _load_images(query_paths)

        return {
            'support_images': support_images,
            'support_labels': torch.tensor(support_labels),
            'query_images': query_images,
            'query_labels': torch.tensor(query_labels),
            'user_id': user_id
        }