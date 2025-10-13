import os
import re
import json
import random
from collections import defaultdict

# Point to the directory containing the signatures of the CEDAR set (after extracting from Kaggle)
# Example: 'D:/datasets/cedar/signatures/full_org'
CEDAR_ORG_DIR = r'C:\Users\USER\.cache\kagglehub\datasets\shreelakshmigp\cedardataset\versions\1\signatures\full_org'
OUTPUT_DIR = 'configs'
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42

def get_user_id_from_filename(filename):
    """Extract user ID from file name (e.g. 'original_1_1.png' -> '1')"""
    match = re.search(r'_(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def create_meta_split():
    print("Start scanning CEDAR dataset directory...")
    if not os.path.isdir(CEDAR_ORG_DIR):
        print(f"ERROR: Directory not found: {CEDAR_ORG_DIR}")
        print("Please update the CEDAR_ORG_DIR variable in the script.")
        return

    user_files = defaultdict(lambda: {'genuine': [], 'forgery': []})
    all_user_ids = set()

    # Scan the 'full_org' directory to get the file list and user IDs
    for filename in os.listdir(CEDAR_ORG_DIR):
        user_id = get_user_id_from_filename(filename)
        if user_id:
            all_user_ids.add(user_id)
            user_files[user_id]['genuine'].append(os.path.join(CEDAR_ORG_DIR.replace('full_org', 'full_org'), filename))

    forg_dir = CEDAR_ORG_DIR.replace('full_org', 'full_forg')
    for filename in os.listdir(forg_dir):
        user_id = get_user_id_from_filename(filename)
        if user_id:
            user_files[user_id]['forgery'].append(os.path.join(forg_dir, filename))

    print(f"Find {len(all_user_ids)} users in dataset.")

    # Split user_ids into train and test
    random.seed(RANDOM_SEED)
    user_id_list = sorted(list(all_user_ids))
    random.shuffle(user_id_list)

    num_test_users = int(len(user_id_list) * TEST_SPLIT_RATIO)
    test_user_ids = user_id_list[:num_test_users]
    train_user_ids = user_id_list[num_test_users:]

    print(f"Split dataset: {len(train_user_ids)} users for meta-train, {len(test_user_ids)} users for meta-test.")

    # Create a dictionary containing file information for each file
    meta_split_data = {
        'meta-train': {uid: user_files[uid] for uid in train_user_ids},
        'meta-test': {uid: user_files[uid] for uid in test_user_ids}
    }

    # Save file JSON
    output_path = os.path.join(OUTPUT_DIR, 'cedar_meta_split.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(meta_split_data, f, indent=4)

    print(f"Saved split file to: {output_path}")

if __name__ == '__main__':
    create_meta_split()