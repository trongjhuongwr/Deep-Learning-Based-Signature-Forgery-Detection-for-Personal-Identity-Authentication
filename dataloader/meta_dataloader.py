import torch
from torch.utils.data import Dataset
import random
import json
from PIL import Image
from torchvision import transforms

def _correct_path(path):
    """
    Convert the path from Windows format in the JSON file
    to the correct format of the Kaggle environment.
    """
    # Replace the first part of the Windows path with the Kaggle path and replace the backslashes
    path = path.replace("C:\\Users\\USER\\.cache\\kagglehub\\datasets\\shreelakshmigp\\cedardataset\\versions\\1", "/kaggle/input/cedardataset")
    return path.replace("\\", "/")

class SignatureEpisodeDataset(Dataset):
    def __init__(self, split_file_path, split_name, k_shot=5, n_query_genuine=5, n_query_forgery=5, transform=None):
        """
        Args:
            split_file_path (str): Path to cedar_meta_split.json file.
            split_name (str): 'meta-train' or 'meta-test'.
            k_shot (int): Number of samples in support set (number of real signatures).
            n_query_genuine (int): Number of genuine query samples.
            n_query_forgery (int): Number of forgery query samples.
            transform (callable, optional): Transform applied to the image.
        """
        with open(split_file_path, 'r') as f:
            self.data = json.load(f)[split_name]

        self.user_ids = list(self.data.keys())
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        
        # Split transform into 2 sets: base and augment
        self.base_transform = transforms.Compose([
            transforms.Resize((220, 150)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.augment_transform = transforms.Compose([
            transforms.Resize((220, 150)),
            transforms.Grayscale(),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Thêm augmentation
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.length = len(self.user_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.data[user_id]

        genuine_paths = [_correct_path(p) for p in user_data['genuine']]
        forgery_paths = [_correct_path(p) for p in user_data['forgery']]

        if len(genuine_paths) < self.k_shot + self.n_query_genuine:
            raise ValueError(f"User {user_id} does not have enough genuine samples. Needs {self.k_shot + self.n_query_genuine}, has {len(genuine_paths)}.")
        if len(forgery_paths) < self.n_query_forgery:
            raise ValueError(f"User {user_id} does not have enough forgery paths. Needs {self.n_query_forgery}, has {len(forgery_paths)}.")

        random.shuffle(genuine_paths)
        random.shuffle(forgery_paths)

        # Create Support Set
        support_paths = genuine_paths[:self.k_shot]

        # Create Query Set
        query_genuine_paths = genuine_paths[self.k_shot : self.k_shot + self.n_query_genuine]
        query_forgery_paths = forgery_paths[:self.n_query_forgery]
        query_paths = query_genuine_paths + query_forgery_paths

        support_labels = [1] * len(support_paths)
        query_labels = [1] * len(query_genuine_paths) + [0] * len(query_forgery_paths)

        # Helper function to load and transform images
        def _load_images(paths, transform_func):
            images = [Image.open(p).convert('L') for p in paths]
            images = [transform_func(img) for img in images]
            return torch.stack(images)

        # Support set uses basic transform 
        support_images = _load_images(support_paths, self.base_transform) 
        # Query set uses transform with augmentation
        query_images = _load_images(query_paths, self.augment_transform)

        return {
            'support_images': support_images,
            'support_labels': torch.tensor(support_labels),
            'query_images': query_images,
            'query_labels': torch.tensor(query_labels),
            'user_id': user_id
        }