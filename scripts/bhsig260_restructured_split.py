import os
import re
import json
import random
from collections import defaultdict
import argparse

def restructure_bhsig_dataset(base_dir, output_dir, num_bengali_test=50, num_hindi_test=30, seed=42):
    """
    Tái cấu trúc bộ dữ liệu BHSig-260 từ định dạng không chuẩn thành 
    một file JSON duy nhất có thể sử dụng cho meta-learning.

    Args:
        base_dir (str): Đường dẫn gốc đến dataset 'cedarbhsig-260'.
        output_dir (str): Thư mục để lưu file JSON kết quả.
        num_bengali_test (int): Số lượng người dùng Bengali để đưa vào tập test.
        num_hindi_test (int): Số lượng người dùng Hindi để đưa vào tập test.
        seed (int): Seed cho việc lấy mẫu ngẫu nhiên để đảm bảo tính tái lặp.
    """
    print("Bắt đầu quá trình tái cấu trúc dataset BHSig-260...")

    # --- CÁC THƯ MỤC NGUỒN ---
    # Các đường dẫn này dựa trên cấu trúc của dataset 'cedarbhsig-260'
    source_dirs = {
        'hindi_genuine': os.path.join(base_dir, 'Reconstructed_BHSig160-20250403T183033Z-001/Reconstructed_BHSig160/Genuine'),
        'hindi_forged': os.path.join(base_dir, 'Reconstructed_BHSig160-20250403T183033Z-001/Reconstructed_BHSig160/Forged'),
        'bengali_genuine': os.path.join(base_dir, 'Reconstructed_BHSig100-20250403T193101Z-001/Reconstructed_BHSig100/Genuine'),
        'bengali_forged': os.path.join(base_dir, 'Reconstructed_BHSig100-20250403T193101Z-001/Reconstructed_BHSig100/Forged')
    }

    # --- HÀM TRÍCH XUẤT THÔNG TIN TỪ TÊN FILE ---
    def parse_filename(filename):
        # Ví dụ: H-S-011-G-01.tif hoặc B-S-011-F-01.tif
        parts = filename.split('-')
        if len(parts) >= 4:
            language = parts[0]
            user_id = int(parts[2])
            return language, user_id
        return None, None

    # --- THU THẬP VÀ PHÂN LOẠI TẤT CẢ CÁC FILE ---
    all_files = defaultdict(lambda: {'genuine': [], 'forgery': []})
    for key, directory in source_dirs.items():
        if not os.path.exists(directory):
            print(f"Cảnh báo: Không tìm thấy thư mục '{directory}'. Bỏ qua.")
            continue
        
        for filename in os.listdir(directory):
            if filename.lower().endswith('.tif'):
                lang, user_id = parse_filename(filename)
                if lang and user_id:
                    full_path = os.path.join(directory, filename)
                    # Tạo một ID người dùng duy nhất bằng cách kết hợp ngôn ngữ và ID
                    unique_user_id = f"{lang}-{user_id}"
                    
                    if 'genuine' in key:
                        all_files[unique_user_id]['genuine'].append(full_path)
                    elif 'forged' in key:
                        all_files[unique_user_id]['forgery'].append(full_path)

    print(f"Đã xử lý và phân loại file cho {len(all_files)} người dùng.")

    # --- CHỌN NGẪU NHIÊN NGƯỜI DÙNG CHO TẬP TEST ---
    bengali_users = sorted([uid for uid in all_files if uid.startswith('B-')])
    hindi_users = sorted([uid for uid in all_files if uid.startswith('H-')])

    random.seed(seed)
    random.shuffle(bengali_users)
    random.shuffle(hindi_users)

    # Lấy ra một tập con để làm meta-test
    test_bengali_ids = bengali_users[:num_bengali_test]
    test_hindi_ids = hindi_users[:num_hindi_test]
    test_user_ids = test_bengali_ids + test_hindi_ids
    
    # --- TẠO FILE JSON CUỐI CÙNG ---
    meta_test_data = {user_id: all_files[user_id] for user_id in test_user_ids}
    final_split_data = {'meta-test': meta_test_data}

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bhsig_restructured_split.json')
    
    with open(output_path, 'w') as f:
        json.dump(final_split_data, f, indent=4)

    print("\n--- HOÀN TẤT TÁI CẤU TRÚC ---")
    print(f"Đã tạo file split cho {len(test_bengali_ids)} người dùng Bengali và {len(test_hindi_ids)} người dùng Hindi.")
    print(f"File được lưu tại: {output_path}")

if __name__ == '__main__':
    # Đoạn mã này cho phép bạn chạy script từ dòng lệnh
    # Ví dụ: python scripts/restructure_bhsig.py --base_dir /kaggle/input/cedarbhsig-260 --output_dir /kaggle/working/splits
    
    parser = argparse.ArgumentParser(description="Restructure BHSig-260 dataset into a meta-learning split file.")
    parser.add_argument('--base_dir', type=str, required=True, help="Đường dẫn gốc đến dataset 'cedarbhsig-260'.")
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/splits', help="Thư mục để lưu file JSON kết quả.")
    parser.add_argument('--num_bengali', type=int, default=50, help="Số người dùng Bengali để đưa vào tập test.")
    parser.add_argument('--num_hindi', type=int, default=30, help="Số người dùng Hindi để đưa vào tập test.")
    parser.add_argument('--seed', type=int, default=42, help="Seed cho việc lấy mẫu ngẫu nhiên.")
    
    args = parser.parse_args()
    
    restructure_bhsig_dataset(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_bengali_test=args.num_bengali,
        num_hindi_test=args.num_hindi,
        seed=args.seed
    )