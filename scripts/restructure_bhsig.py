import os
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

def get_bhsig_info_from_path(filepath: str) -> Optional[Tuple[str, int]]:
    """
    Extracts language code ('B' or 'H') and user ID from BHSig file paths
    within the nth2165 dataset structure.
    Example path: .../BHSig260/Bengali/001/B-S-001-G-01.tif

    Args:
        filepath (str): The absolute path to the signature file.

    Returns:
        Optional[Tuple[str, int]]: ('B' or 'H', user_id) or None if parsing fails.
    """
    try:
        parts = filepath.split(os.sep)
        filename = parts[-1]
        user_id_str = parts[-2] # User ID is the parent directory name
        language_folder = parts[-3] # 'Bengali' or 'Hindi'

        # Extract language code from filename prefix or folder name
        language_code = None
        if filename.upper().startswith('B-'):
            language_code = 'B'
        elif filename.upper().startswith('H-'):
            language_code = 'H'
        elif language_folder.lower() == 'bengali':
             language_code = 'B'
        elif language_folder.lower() == 'hindi':
             language_code = 'H'

        if language_code and user_id_str.isdigit():
            user_id = int(user_id_str)
            return language_code, user_id
    except (IndexError, ValueError):
        # Handle cases where the path structure is unexpected
        pass
    # print(f"Warning: Could not parse language/user ID from path: {filepath}")
    return None

def restructure_nth2165_bhsig(base_dir: str, output_dir: str, num_bengali_test: int = 50, num_hindi_test: int = 30, seed: int = 42):
    """
    Scans the 'nth2165/bhsig260-hindi-bengali' dataset, separates Bengali and Hindi users,
    and creates two separate JSON files ('bhsig_bengali_meta_test.json', 'bhsig_hindi_meta_test.json')
    containing absolute paths for randomly selected test users.

    Args:
        base_dir (str): Path to the root directory of the 'nth2165/bhsig260-hindi-bengali' dataset
                        (containing 'Bengali' and 'Hindi' subdirectories).
        output_dir (str): Directory where the restructured JSON split files will be saved.
        num_bengali_test (int): Number of Bengali users to include in the Bengali test split. Defaults to 50.
        num_hindi_test (int): Number of Hindi users to include in the Hindi test split. Defaults to 30.
        seed (int): Random seed for selecting test users. Defaults to 42.
    """
    print("--- Starting BHSig-260 (nth2165) Dataset Restructuring ---")

    bengali_dir = os.path.join(base_dir, 'Bengali')
    hindi_dir = os.path.join(base_dir, 'Hindi')

    # --- 1. Validate Input Directories ---
    if not os.path.isdir(bengali_dir):
        print(f"ERROR: Bengali directory not found: {bengali_dir}")
        return
    if not os.path.isdir(hindi_dir):
        print(f"ERROR: Hindi directory not found: {hindi_dir}")
        return

    # --- 2. Collect All Files Grouped by Unique User ID ('B-1', 'H-42', etc.) ---
    all_files: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {'genuine': [], 'forgery': []})
    processed_files_count = 0
    skipped_files_count = 0
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    print("Scanning Bengali and Hindi directories...")
    for lang_dir in [bengali_dir, hindi_dir]:
        print(f"  Processing directory: {lang_dir}")
        for user_id_folder in os.listdir(lang_dir):
            user_dir_path = os.path.join(lang_dir, user_id_folder)
            if os.path.isdir(user_dir_path) and user_id_folder.isdigit():
                user_id = int(user_id_folder)
                lang_code = 'B' if 'Bengali' in lang_dir else 'H'
                unique_user_id = f"{lang_code}-{user_id}"

                for filename in os.listdir(user_dir_path):
                    if filename.lower().endswith(supported_extensions):
                        full_path = os.path.join(user_dir_path, filename)
                        # Determine genuine/forged based on filename convention (B-S- or B-F-)
                        if filename.upper().startswith(f'{lang_code}-S-'):
                            all_files[unique_user_id]['genuine'].append(full_path)
                        elif filename.upper().startswith(f'{lang_code}-F-'):
                            all_files[unique_user_id]['forgery'].append(full_path)
                        else:
                            # print(f"    Skipping file with unclear type: {filename}")
                            skipped_files_count += 1
                            continue # Skip files not matching S/F pattern
                        processed_files_count += 1
                    else:
                        skipped_files_count += 1 # Skip non-image files

    if not all_files:
         print("ERROR: No valid BHSig files found or parsed. Cannot create splits.")
         return

    print(f"Finished scanning. Processed {processed_files_count} files, skipped {skipped_files_count}.")
    print(f"Found data for {len(all_files)} unique users (Bengali + Hindi).")

    # --- 3. Separate Users by Language and Select Test Subsets ---
    bengali_users = sorted([uid for uid in all_files if uid.startswith('B-')])
    hindi_users = sorted([uid for uid in all_files if uid.startswith('H-')])

    print(f"  Available Bengali users: {len(bengali_users)}")
    print(f"  Available Hindi users: {len(hindi_users)}")

    # Adjust requested numbers if fewer users are available
    num_bengali_test = min(num_bengali_test, len(bengali_users))
    num_hindi_test = min(num_hindi_test, len(hindi_users))

    random.seed(seed)
    random.shuffle(bengali_users)
    random.shuffle(hindi_users)

    test_bengali_ids = bengali_users[:num_bengali_test]
    test_hindi_ids = hindi_users[:num_hindi_test]

    print(f"Selected {len(test_bengali_ids)} Bengali users for Bengali test split.")
    print(f"Selected {len(test_hindi_ids)} Hindi users for Hindi test split.")

    # --- 4. Create and Save Separate JSON Files ---
    os.makedirs(output_dir, exist_ok=True)

    # Save Bengali Split
    if test_bengali_ids:
        bengali_meta_test_data = {uid: all_files[uid] for uid in test_bengali_ids}
        bengali_split_data = {'meta-test': bengali_meta_test_data}
        bengali_output_path = os.path.join(output_dir, 'bhsig_bengali_meta_test.json')
        try:
            with open(bengali_output_path, 'w', encoding='utf-8') as f:
                json.dump(bengali_split_data, f, indent=4)
            print(f"  Successfully saved Bengali meta-test split to: {bengali_output_path}")
        except Exception as e:
            print(f"  ERROR saving Bengali split file: {e}")
    else:
        print("  No Bengali users selected, skipping Bengali split file generation.")


    # Save Hindi Split
    if test_hindi_ids:
        hindi_meta_test_data = {uid: all_files[uid] for uid in test_hindi_ids}
        hindi_split_data = {'meta-test': hindi_meta_test_data}
        hindi_output_path = os.path.join(output_dir, 'bhsig_hindi_meta_test.json')
        try:
            with open(hindi_output_path, 'w', encoding='utf-8') as f:
                json.dump(hindi_split_data, f, indent=4)
            print(f"  Successfully saved Hindi meta-test split to: {hindi_output_path}")
        except Exception as e:
            print(f"  ERROR saving Hindi split file: {e}")
    else:
        print("  No Hindi users selected, skipping Hindi split file generation.")

    print("--- BHSig-260 Restructuring Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Restructure the 'nth2165/bhsig260-hindi-bengali' dataset "
                                                 "into separate Bengali and Hindi meta-test JSON files.")
    parser.add_argument('--base_dir', type=str, required=True,
                        help="Path to the root directory of the 'nth2165/bhsig260-hindi-bengali' dataset.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the output JSON split files.")
    parser.add_argument('--num_bengali', type=int, default=50,
                        help="Number of Bengali users for the Bengali 'meta-test' split.")
    parser.add_argument('--num_hindi', type=int, default=30,
                        help="Number of Hindi users for the Hindi 'meta-test' split.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for selecting test users.")

    args = parser.parse_args()
    restructure_nth2165_bhsig(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_bengali_test=args.num_bengali,
        num_hindi_test=args.num_hindi,
        seed=args.seed
    )