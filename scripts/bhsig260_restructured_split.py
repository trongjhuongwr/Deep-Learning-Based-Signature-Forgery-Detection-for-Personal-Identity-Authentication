import os
import re
import json
import random
from collections import defaultdict

print("Bắt đầu quá trình tái cấu trúc dataset cedarbhsig-260...")

# --- CẤU HÌNH ---
# Đường dẫn gốc tới dataset bạn đã tìm thấy
BASE_BHSIG_DIR = '/kaggle/input/cedarbhsig-260/'
OUTPUT_FILE = '/kaggle/working/bhsig_restructured_split.json'
NUM_BENGALI_TEST_USERS = 50  # Số người dùng Bengali để test
NUM_HINDI_TEST_USERS = 30    # Số người dùng Hindi để test
RANDOM_SEED = 42

# --- CÁC THƯ MỤC NGUỒN ---
# Lưu ý: Các đường dẫn này dựa trên mô tả của bạn. Hãy kiểm tra lại nếu cần.
HINDI_GENUINE_DIR = os.path.join(BASE_BHSIG_DIR, 'Reconstructed_BHSig160-20250403T183033Z-001/Reconstructed_BHSig160/Genuine')
HINDI_FORGED_DIR = os.path.join(BASE_BHSIG_DIR, 'Reconstructed_BHSig160-20250403T183033Z-001/Reconstructed_BHSig160/Forged')
BENGALI_GENUINE_DIR = os.path.join(BASE_BHSIG_DIR, 'Reconstructed_BHSig100-20250403T193101Z-001/Reconstructed_BHSig100/Genuine')
BENGALI_FORGED_DIR = os.path.join(BASE_BHSIG_DIR, 'Reconstructed_BHSig100-20250403T193101Z-001/Reconstructed_BHSig100/Forged')

# --- HÀM TRÍCH XUẤT THÔNG TIN TỪ TÊN FILE ---
def parse_filename(filename):
    # Ví dụ: H-S-011-G-01.tif hoặc B-S-011-F-01.tif
    parts = filename.split('-')
    if len(parts) >= 4:
        language = parts[0]  # 'H' hoặc 'B'
        user_id = int(parts[2])
        return language, user_id
    return None, None

# --- THU THẬP VÀ PHÂN LOẠI FILE ---
all_files = defaultdict(lambda: {'genuine': [], 'forgery': []})
source_dirs = {
    'hindi_genuine': HINDI_GENUINE_DIR,
    'hindi_forged': HINDI_FORGED_DIR,
    'bengali_genuine': BENGALI_GENUINE_DIR,
    'bengali_forged': BENGALI_FORGED_DIR
}

for key, directory in source_dirs.items():
    if not os.path.exists(directory):
        print(f"Cảnh báo: Không tìm thấy thư mục '{directory}'. Bỏ qua.")
        continue
        
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
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

# --- CHỌN NGẪU NHIÊN NGƯỜI DÙNG ĐỂ TEST ---
bengali_users = [uid for uid in all_files if uid.startswith('B-')]
hindi_users = [uid for uid in all_files if uid.startswith('H-')]

random.seed(RANDOM_SEED)
random.shuffle(bengali_users)
random.shuffle(hindi_users)

test_bengali_ids = bengali_users[:NUM_BENGALI_TEST_USERS]
test_hindi_ids = hindi_users[:NUM_HINDI_TEST_USERS]

# --- TẠO FILE JSON CUỐI CÙNG ---
meta_test_data = {}
for user_id in test_bengali_ids + test_hindi_ids:
    meta_test_data[user_id] = all_files[user_id]

final_split_data = {'meta-test': meta_test_data}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(final_split_data, f, indent=4)

print("\n--- HOÀN TẤT TÁI CẤU TRÚC ---")
print(f"Đã tạo file split cho {len(test_bengali_ids)} người dùng Bengali và {len(test_hindi_ids)} người dùng Hindi.")
print(f"File được lưu tại: {OUTPUT_FILE}")