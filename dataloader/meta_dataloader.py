# Dán toàn bộ nội dung này để thay thế cho file: dataloader/meta_dataloader.py

import torch
from torch.utils.data import Dataset
import random
import json
from PIL import Image
import os
from torchvision import transforms

class SignatureEpisodeDataset(Dataset):
    def __init__(self, split_file_path, base_data_dir, split_name, k_shot=5, n_query_genuine=5, n_query_forgery=5, augment=False, use_full_path=False):
        """
        Args:
            split_file_path (str): Đường dẫn đến file JSON chứa thông tin split.
            base_data_dir (str): Đường dẫn gốc tới thư mục dataset (chỉ dùng khi use_full_path=False).
            split_name (str): 'meta-train' hoặc 'meta-test'.
            k_shot (int): Số mẫu trong support set.
            n_query_genuine (int): Số mẫu thật trong query set.
            n_query_forgery (int): Số mẫu giả trong query set.
            transform (callable, optional): Transform áp dụng cho ảnh.
            use_full_path (bool): Nếu True, sử dụng đường dẫn tuyệt đối từ file JSON.
                                  Nếu False, nối base_data_dir với đường dẫn tương đối.
            augment (bool): Nếu True, áp dụng data augmentation.
        """
        with open(split_file_path, 'r') as f:
            split_content = json.load(f)[split_name]
        
        self.data = {}
        if use_full_path:
            self.data = split_content
        else:
            if base_data_dir is None:
                raise ValueError("base_data_dir phải được cung cấp khi use_full_path=False")
            for user_id, paths in split_content.items():
                self.data[user_id] = {
                    'genuine': [os.path.join(base_data_dir, p) for p in paths['genuine']],
                    'forgery': [os.path.join(base_data_dir, p) for p in paths['forgery']]
                }

        self.user_ids = list(self.data.keys())
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.length = len(self.user_ids)

        # Định nghĩa 2 bộ transform
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

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.data[user_id]

        genuine_paths = user_data['genuine']
        forgery_paths = user_data['forgery']

        # Xử lý nếu không đủ mẫu
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            genuine_paths = random.choices(genuine_paths, k=required_genuine)
        
        if len(forgery_paths) < self.n_query_forgery:
            forgery_paths = random.choices(forgery_paths, k=self.n_query_forgery)

        random.shuffle(genuine_paths)
        random.shuffle(forgery_paths)

        support_paths = genuine_paths[:self.k_shot]
        query_genuine_paths = genuine_paths[self.k_shot : self.k_shot + self.n_query_genuine]
        query_forgery_paths = forgery_paths[:self.n_query_forgery]
        
        query_paths = query_genuine_paths + query_forgery_paths
        query_labels = [1] * len(query_genuine_paths) + [0] * len(query_forgery_paths)

        # Chọn transform phù hợp
        transform_to_apply = self.augment_transform if self.augment else self.base_transform

        def _load_images(paths):
            images = [Image.open(p).convert('L') for p in paths]
            images = [transform_to_apply(img) for img in images]
            return torch.stack(images)

        support_images = _load_images(support_paths)
        query_images = _load_images(query_paths)

        return {
            'support_images': support_images,
            'query_images': query_images,
            'query_labels': torch.tensor(query_labels),
            'user_id': user_id
        }