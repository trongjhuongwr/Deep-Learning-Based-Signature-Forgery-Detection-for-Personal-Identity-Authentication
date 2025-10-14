import torch
from torch.utils.data import Dataset
import random
import json
from PIL import Image
from torchvision import transforms
import os

class SignatureEpisodeDataset(Dataset):
    def __init__(self, split_file_path, base_data_dir, split_name, k_shot=5, n_query_genuine=5, n_query_forgery=5):
        """
        Args:
            split_file_path (str): Đường dẫn đến file cedar_meta_split.json.
            base_data_dir (str): Đường dẫn gốc đến thư mục dataset trên Kaggle (ví dụ: '/kaggle/input/cedardataset').
            split_name (str): 'meta-train' hoặc 'meta-test'.
            ... (các tham số khác giữ nguyên)
        """
        with open(split_file_path, 'r') as f:
            self.split_data = json.load(f)[split_name]

        self.base_data_dir = base_data_dir
        self.user_ids = list(self.split_data.keys())
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery

        # Tách transform thành 2 bộ: base và augment
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
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.length = len(self.user_ids)

    def _get_kaggle_path(self, windows_path):
        # Chuyển đổi đường dẫn từ Windows (lưu trong JSON) sang đường dẫn Kaggle
        # Ví dụ: 'C:\\...\\signatures\\full_org\\original_1_1.png' -> '/kaggle/input/cedardataset/signatures/full_org/original_1_1.png'
        parts = windows_path.replace('\\', '/').split('/')
        # Tìm vị trí của 'signatures' và lấy phần còn lại
        try:
            sig_index = parts.index('signatures')
            relative_path = '/'.join(parts[sig_index:])
            return os.path.join(self.base_data_dir, relative_path)
        except ValueError:
            # Nếu không tìm thấy, trả về đường dẫn gốc, có thể gây lỗi nhưng tránh crash
            return windows_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.split_data[user_id]

        genuine_paths = [self._get_kaggle_path(p) for p in user_data['genuine']]
        forgery_paths = [self._get_kaggle_path(p) for p in user_data['forgery']]

        # ... (Phần còn lại của hàm __getitem__ giữ nguyên như cũ) ...
        if len(genuine_paths) < self.k_shot + self.n_query_genuine:
            raise ValueError(f"User {user_id} không có đủ mẫu thật...")
        if len(forgery_paths) < self.n_query_forgery:
            raise ValueError(f"User {user_id} không có đủ mẫu giả...")

        random.shuffle(genuine_paths)
        random.shuffle(forgery_paths)

        support_paths = genuine_paths[:self.k_shot]
        query_genuine_paths = genuine_paths[self.k_shot : self.k_shot + self.n_query_genuine]
        query_forgery_paths = forgery_paths[:self.n_query_forgery]

        query_paths = query_genuine_paths + query_forgery_paths
        support_labels = [1] * len(support_paths)
        query_labels = [1] * len(query_genuine_paths) + [0] * len(query_forgery_paths)

        def _load_images(paths, transform_func):
            images = [Image.open(p).convert('L') for p in paths]
            images = [transform_func(img) for img in images]
            return torch.stack(images)

        support_images = _load_images(support_paths, self.base_transform)
        query_images = _load_images(query_paths, self.augment_transform)

        return {
            'support_images': support_images,
            'support_labels': torch.tensor(support_labels),
            'query_images': query_images,
            'query_labels': torch.tensor(query_labels),
            'user_id': user_id
        }