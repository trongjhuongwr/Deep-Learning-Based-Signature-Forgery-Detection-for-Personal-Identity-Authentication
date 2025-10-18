import os
import re
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

def parse_bhsig_filename(filename: str) -> Optional[Tuple[str, int]]:
    """
    Extracts language ('B' or 'H') and user ID from BHSig-style filenames.
    Example: 'B-S-011-G-01.tif' -> ('B', 11)
             'H-S-042-F-03.tif' -> ('H', 42)

    Args:
        filename (str): The input filename.

    Returns:
        Optional[Tuple[str, int]]: A tuple (language_code, user_id), or None if parsing fails.
    """
    # Regex to capture Language (B or H) and User ID (digits)
    # Allows for variations like -S-, -G-, -F-
    match = re.match(r'^([BH])-[SFG]-(\d+)-[GF]-(\d+)\.tif$', filename, re.IGNORECASE)
    if match:
        language_code = match.group(1).upper() # 'B' or 'H'
        user_id = int(match.group(2))
        return language_code, user_id
    else:
        # print(f"Warning: Could not parse BHSig filename: {filename}")
        return None

def restructure_bhsig_dataset(base_dir: str, output_dir: str, num_bengali_test: int = 50, num_hindi_test: int = 30, seed: int = 42):
    """
    Restructures the potentially mixed 'cedarbhsig-260' dataset into a single
    JSON file suitable for meta-learning evaluation, containing absolute paths.
    Selects a random subset of Bengali and Hindi users for the 'meta-test' split.

    Args:
        base_dir (str): Path to the root directory of the 'cedarbhsig-260' dataset.
        output_dir (str): Directory where the restructured 'meta-test' split JSON file will be saved.
        num_bengali_test (int): Number of Bengali users to include in the test split. Defaults to 50.
        num_hindi_test (int): Number of Hindi users to include in the test split. Defaults to 30.
        seed (int): Random seed for selecting test users to ensure reproducibility. Defaults to 42.
    """
    print("--- Starting BHSig-260 Dataset Restructuring ---")

    # --- Define Source Directories based on the expected structure ---
    # These paths are based on the structure observed in the 'cedarbhsig-260' dataset
    source_dirs = {
        # Note: Directory names seem inconsistent (160 vs 100). Verify these paths.
        'hindi_genuine': os.path.join(base_dir, 'Reconstructed_BHSig160-20250403T183033Z-001/Reconstructed_BHSig160/Genuine'),
        'hindi_forged': os.path.join(base_dir, 'Reconstructed_BHSig160-20250403T183033Z-001/Reconstructed_BHSig160/Forged'),
        'bengali_genuine': os.path.join(base_dir, 'Reconstructed_BHSig100-20250403T193101Z-001/Reconstructed_BHSig100/Genuine'),
        'bengali_forged': os.path.join(base_dir, 'Reconstructed_BHSig100-20250403T193101Z-001/Reconstructed_BHSig100/Forged')
    }

    # --- 1. Collect and Classify All Files ---
    # Store absolute paths grouped by a unique user ID (e.g., 'B-11', 'H-42')
    all_files: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {'genuine': [], 'forgery': []})
    processed_files_count = 0
    skipped_files_count = 0

    print("Scanning source directories...")
    for key, directory in source_dirs.items():
        if not os.path.exists(directory):
            print(f"Warning: Source directory not found: '{directory}'. Skipping.")
            continue

        print(f"  Processing: {directory}")
        for filename in os.listdir(directory):
            # Process only .tif files (case-insensitive)
            if filename.lower().endswith('.tif'):
                parsed_info = parse_bhsig_filename(filename)
                if parsed_info:
                    lang, user_id = parsed_info
                    unique_user_id = f"{lang}-{user_id}" # Create unique ID
                    full_path = os.path.join(directory, filename)

                    # Determine if genuine or forged based on the source directory key
                    if 'genuine' in key:
                        all_files[unique_user_id]['genuine'].append(full_path)
                    elif 'forged' in key:
                        all_files[unique_user_id]['forgery'].append(full_path)
                    processed_files_count += 1
                else:
                    # print(f"    Skipping file with unparsable name: {filename}")
                    skipped_files_count += 1
            else:
                skipped_files_count += 1 # Skip non-tif files silently


    if not all_files:
         print("ERROR: No valid BHSig files found or parsed. Cannot create split.")
         return

    print(f"Finished scanning. Processed {processed_files_count} files, skipped {skipped_files_count}.")
    print(f"Found data for {len(all_files)} unique users (Bengali + Hindi).")

    # --- 2. Select Random Subset of Users for Meta-Test ---
    bengali_users = sorted([uid for uid in all_files if uid.startswith('B-')])
    hindi_users = sorted([uid for uid in all_files if uid.startswith('H-')])

    print(f"  Available Bengali users: {len(bengali_users)}")
    print(f"  Available Hindi users: {len(hindi_users)}")


    # Adjust requested numbers if fewer users are available
    num_bengali_test = min(num_bengali_test, len(bengali_users))
    num_hindi_test = min(num_hindi_test, len(hindi_users))

    # Set seed for random sampling
    random.seed(seed)
    # Shuffle lists in place
    random.shuffle(bengali_users)
    random.shuffle(hindi_users)

    # Select the test users
    test_bengali_ids = bengali_users[:num_bengali_test]
    test_hindi_ids = hindi_users[:num_hindi_test]
    test_user_ids = test_bengali_ids + test_hindi_ids

    if not test_user_ids:
         print("ERROR: No test users selected. Cannot create split file.")
         return

    print(f"Selected {len(test_bengali_ids)} Bengali and {len(test_hindi_ids)} Hindi users for the meta-test split.")

    # --- 3. Create the Final JSON Structure ---
    meta_test_data = {
        user_id: all_files[user_id] for user_id in test_user_ids if user_id in all_files
    }

    # Final structure expected by the dataloader
    final_split_data = {'meta-test': meta_test_data}

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'bhsig_restructured_split.json')

    # --- 4. Save the JSON File ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_split_data, f, indent=4)
        print("\n--- BHSig-260 Restructuring Complete ---")
        print(f"Successfully created meta-test split file with {len(test_user_ids)} users.")
        print(f"File saved to: {output_path}")
    except IOError as e:
        print(f"ERROR: Could not write output JSON file: {e}")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while saving the JSON file: {e}")


if __name__ == '__main__':
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Restructure the BHSig-260 dataset from the 'cedarbhsig-260' source "
                                                 "into a meta-learning compatible JSON split file.")
    parser.add_argument('--base_dir', type=str, required=True,
                        help="Path to the root directory of the 'cedarbhsig-260' dataset.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save the output 'bhsig_restructured_split.json' file.")
    parser.add_argument('--num_bengali', type=int, default=50,
                        help="Number of Bengali users to include in the 'meta-test' split.")
    parser.add_argument('--num_hindi', type=int, default=30,
                        help="Number of Hindi users to include in the 'meta-test' split.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for selecting test users.")

    args = parser.parse_args()

    # Execute the restructuring function
    restructure_bhsig_dataset(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_bengali_test=args.num_bengali,
        num_hindi_test=args.num_hindi,
        seed=args.seed
    )