#!/usr/bin/env python3
"""
Add cell GT information from original annotations to JSONL files
"""

import json
from pathlib import Path
import argparse
from tqdm import tqdm


def add_cell_gt(
    input_jsonl_path: str,
    output_jsonl_path: str,
    annotations_dir: str
):
    """Add cell GT bboxes from original annotations to JSONL entries"""
    
    print(f"Loading JSONL from: {input_jsonl_path}")
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]
    
    print(f"Found {len(entries)} entries")
    
    # Process each entry
    updated_entries = []
    missing_annotations = []
    
    for entry in tqdm(entries, desc="Adding cell GT"):
        # Extract image name from filename
        filename = entry['filename']
        img_name = Path(filename).stem  # e.g., 'fintabnet_val_000009'
        
        # Find corresponding annotation file
        annotation_path = Path(annotations_dir) / f"{img_name}.json"
        
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                annotation = json.load(f)
            
            # Add cell GT
            entry['cell_gt'] = annotation.get('cell_bboxes', [])
            entry['cell_count'] = len(entry['cell_gt'])
        else:
            missing_annotations.append(filename)
            entry['cell_gt'] = []
            entry['cell_count'] = 0
        
        updated_entries.append(entry)
    
    # Save updated entries
    print(f"\nSaving updated JSONL to: {output_jsonl_path}")
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in updated_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print(f"Total entries processed: {len(entries)}")
    print(f"Entries with cell GT added: {len(entries) - len(missing_annotations)}")
    print(f"Missing annotations: {len(missing_annotations)}")
    
    if missing_annotations:
        print("\nFirst 10 missing annotations:")
        for filename in missing_annotations[:10]:
            print(f"  {filename}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Add cell GT to JSONL files")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="/home/hongjunchoi/local_data/tsr/fintabnet/annotations/val",
        help="Directory containing annotation JSON files"
    )
    
    args = parser.parse_args()
    
    add_cell_gt(
        input_jsonl_path=args.input_jsonl,
        output_jsonl_path=args.output_jsonl,
        annotations_dir=args.annotations_dir
    )


if __name__ == "__main__":
    main()