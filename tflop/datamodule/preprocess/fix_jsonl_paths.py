#!/usr/bin/env python3
"""
Fix file_name paths in existing JSONL files
"""

import json
import os
from tqdm import tqdm
import argparse

def fix_file_path(file_name, split, dataset):
    """
    Convert file_name to correct path structure
    From: marketing_image_000002_1634629424.103172.png
    To: marketing/images/val/image_000002_1634629424.103172.png
    """
    # Remove dataset prefix if it exists
    if file_name.startswith(f"{dataset}_"):
        filename = file_name[len(f"{dataset}_"):]
    else:
        filename = file_name

    # Map split names
    if split == 'val':
        split_dir = 'val'
    elif split == 'test':
        split_dir = 'test'
    else:
        split_dir = 'train'

    # Create correct path
    return f"{dataset}/images/{split_dir}/{filename}"

def fix_gold_coord(gold_coord):
    """Convert gold_coord from dict to list of strings format"""
    if not gold_coord:
        return None

    # If already a list, check if it needs conversion
    if isinstance(gold_coord, list):
        return gold_coord

    # Convert from dict to list of strings
    gold_coord_list = []
    for cell_id in sorted(gold_coord.keys(), key=lambda x: int(x)):
        cell_data = gold_coord[cell_id]

        # Extract bbox
        if isinstance(cell_data, list) and len(cell_data) > 0:
            bbox = cell_data[0][0] if isinstance(cell_data[0], list) else [0, 0, 0, 0]
            text = cell_data[2] if len(cell_data) > 2 else ""
        else:
            bbox = [0, 0, 0, 0]
            text = ""

        # Type: 2 for filled cells, 1 for empty cells
        cell_type = 2 if text and bbox != [0, 0, 0, 0] else 1

        # Format: "x1 y1 x2 y2 type"
        coord_str = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {cell_type}"
        gold_coord_list.append(coord_str)

    return gold_coord_list

def fix_jsonl_file(input_file, output_file):
    """Fix paths and gold_coord in a single JSONL file"""

    print(f"Processing {input_file}...")

    # Read all lines
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Process and write fixed lines
    with open(output_file, 'w') as f:
        for line in tqdm(lines, desc=f"Fixing {os.path.basename(input_file)}"):
            data = json.loads(line.strip())

            # Fix the file_name path
            if 'file_name' in data and 'split' in data and 'dataset' in data:
                data['file_name'] = fix_file_path(
                    data.get('original_file', data['file_name']),
                    data['split'],
                    data['dataset']
                )

            # Fix gold_coord format
            if 'gold_coord' in data:
                data['gold_coord'] = fix_gold_coord(data['gold_coord'])

            f.write(json.dumps(data) + '\n')

    print(f"Saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Fix file paths in JSONL files')
    parser.add_argument('--input_dir', type=str,
                       default='/home/hongjunchoi/local_data/tsr/synthetic/meta_data',
                       help='Directory containing JSONL files to fix')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory (default: same as input_dir)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Fix each JSONL file
    jsonl_files = ['dataset_train.jsonl', 'dataset_validation.jsonl', 'dataset_test.jsonl']

    for jsonl_file in jsonl_files:
        input_path = os.path.join(args.input_dir, jsonl_file)

        if not os.path.exists(input_path):
            print(f"Skipping {jsonl_file} (not found)")
            continue

        # Create backup name
        output_path = os.path.join(args.output_dir, jsonl_file.replace('.jsonl', '_fixed.jsonl'))

        fix_jsonl_file(input_path, output_path)

    print("\nAll files fixed! Original files are preserved.")
    print(f"Fixed files have '_fixed' suffix.")

if __name__ == '__main__':
    main()