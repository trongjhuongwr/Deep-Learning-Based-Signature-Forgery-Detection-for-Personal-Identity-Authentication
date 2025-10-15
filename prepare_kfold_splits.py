import json
import os
import random
from sklearn.model_selection import KFold
from collections import defaultdict
import re

# --- CẤU HÌNH ---
# Đường dẫn đến thư mục chứa chữ ký gốc của bộ CEDAR
# LƯU Ý: Hãy đảm bảo đường dẫn này đúng với môi trường Kaggle của bạn
BASE_DATA_DIR = '/kaggle/input/cedardataset/signatures'
OUTPUT_DIR = '/kaggle/working/kfold_splits' # Sẽ lưu các file split vào thư mục working
NUM_SPLITS = 5
RANDOM_SEED = 42

# --- CÁC HÀM HỖ TRỢ (Lấy từ file cũ của bạn) ---
def get_user_id_from_filename(filename):
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

# --- CHỨC NĂNG CHÍNH ---
def create_kfold_splits():
    print("Start scanning CEDAR data folder...")
    org_dir = os.path.join(BASE_DATA_DIR, 'full_org')
    forg_dir = os.path.join(BASE_DATA_DIR, 'full_forg')

    if not os.path.isdir(org_dir):
        print(f"ERROR: Directory not found: {org_dir}")
        return

    # 1. Thu thập tất cả các file cho từng người dùng
    user_files = defaultdict(lambda: {'genuine': [], 'forgery': []})
    all_user_ids = set()

    for filename in os.listdir(org_dir):
        user_id = get_user_id_from_filename(filename)
        if user_id:
            all_user_ids.add(user_id)
            user_files[user_id]['genuine'].append(os.path.join(org_dir, filename))

    for filename in os.listdir(forg_dir):
        user_id = get_user_id_from_filename(filename)
        if user_id:
            user_files[user_id]['forgery'].append(os.path.join(forg_dir, filename))
    
    user_id_list = sorted(list(all_user_ids))
    print(f"Found {len(user_id_list)} users in the dataset.")

    # 2. Chia user IDs thành 5 folds
    random.seed(RANDOM_SEED)
    random.shuffle(user_id_list) # Xáo trộn danh sách user IDs
    
    kf = KFold(n_splits=NUM_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Tạo và lưu 5 file JSON cho từng fold
    for fold_idx, (train_indices, test_indices) in enumerate(kf.split(user_id_list)):
        train_user_ids = [user_id_list[i] for i in train_indices]
        test_user_ids = [user_id_list[i] for i in test_indices]
        
        # Tạo dữ liệu split cho fold hiện tại
        split_data = {
            'meta-train': {uid: user_files[uid] for uid in train_user_ids},
            'meta-test': {uid: user_files[uid] for uid in test_user_ids}
        }
        
        # Lưu ra file JSON
        file_path = os.path.join(OUTPUT_DIR, f'cedar_meta_split_fold_{fold_idx}.json')
        with open(file_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        
        print(f"Created files for Fold {fold_idx}: {len(train_user_ids)} train users, {len(test_user_ids)} test users. Saved at: {file_path}")

if __name__ == '__main__':
    create_kfold_splits()