#!/usr/bin/env python3
"""
Unified FinTabNet preprocessing script for TFLOP
Processes original FinTabNet data to TFLOP-compatible format
"""

import json
import os
import argparse
import re
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add TFLOP to path for PubTabNet utils (if needed)
sys.path.append('/home/work/hongjunchoi/git_repo/TFLOP')


def convert_fintabnet_html_to_otsl(html_string: str, otsl_tag_maps: Dict[str, str]) -> Tuple[List[str], int, int]:
    """
    Convert FinTabNet HTML to OTSL sequence with proper span handling
    
    Args:
        html_string: HTML string 
        otsl_tag_maps: Mapping of OTSL tags
        
    Returns:
        otsl_seq: OTSL sequence
        num_rows: Number of rows
        num_cols: Number of columns
    """
    
    # Count rows
    num_rows = html_string.count('<tr>')
    if num_rows == 0:
        return [], 0, 0
    
    # Parse HTML to extract cell information
    rows_html = html_string.split('<tr>')[1:]  # Skip before first <tr>
    
    # First pass: collect all cells with their positions and spans
    all_cells = []
    
    for row_idx, row_html in enumerate(rows_html):
        if '</tr>' in row_html:
            row_html = row_html[:row_html.index('</tr>')]
        
        # Find all td tags in this row
        td_tags = re.findall(r'<td[^>]*>', row_html)
        
        for td_tag in td_tags:
            colspan = 1
            rowspan = 1
            
            # Extract colspan
            colspan_match = re.search(r'colspan="(\d+)"', td_tag)
            if colspan_match:
                colspan = int(colspan_match.group(1))
            
            # Extract rowspan
            rowspan_match = re.search(r'rowspan="(\d+)"', td_tag)
            if rowspan_match:
                rowspan = int(rowspan_match.group(1))
            
            all_cells.append({
                'row': row_idx,
                'colspan': colspan,
                'rowspan': rowspan
            })
    
    # Second pass: build the grid with proper column positioning
    # Track which columns are occupied by rowspans from previous rows
    rowspan_occupancy = {}  # (row, col) -> True if occupied
    
    # Place cells in grid
    cell_positions = []
    
    for row_idx in range(num_rows):
        col_idx = 0
        row_cells = [c for c in all_cells if c['row'] == row_idx]
        
        for cell in row_cells:
            # Skip columns occupied by rowspans from previous rows
            while (row_idx, col_idx) in rowspan_occupancy:
                col_idx += 1
            
            # Record this cell's position
            cell['col'] = col_idx
            cell_positions.append(cell)
            
            # Mark grid positions occupied by this cell
            for r in range(cell['rowspan']):
                for c in range(cell['colspan']):
                    if r > 0 or c > 0:  # Don't mark the origin cell
                        rowspan_occupancy[(row_idx + r, col_idx + c)] = True
            
            # Move to next column position
            col_idx += cell['colspan']
    
    # Determine number of columns
    num_cols = max((cell['col'] + cell['colspan'] for cell in cell_positions), default=0)
    
    if num_cols == 0:
        return [], 0, 0
    
    # Build OTSL grid
    # 0: empty, 1: C-tag, 2: L-tag, 3: U-tag, 4: X-tag
    grid = np.zeros((num_rows, num_cols), dtype=int)
    
    # Fill grid based on cell positions and spans
    for cell in cell_positions:
        row = cell['row']
        col = cell['col']
        colspan = cell['colspan']
        rowspan = cell['rowspan']
        
        # First cell is always C-tag
        grid[row, col] = 1
        
        # Colspan continuation (L-tags)
        for c in range(1, colspan):
            if col + c < num_cols:
                grid[row, col + c] = 2
        
        # Rowspan continuation (U-tags)
        for r in range(1, rowspan):
            if row + r < num_rows:
                # First column of rowspan
                grid[row + r, col] = 3
                
                # Rest of columns in rowspan (also U-tags for multi-col spans)
                for c in range(1, colspan):
                    if col + c < num_cols:
                        grid[row + r, col + c] = 3
    
    # Convert grid to OTSL sequence
    otsl_seq = []
    
    for r in range(num_rows):
        for c in range(num_cols):
            cell_type = grid[r, c]
            
            if cell_type == 0 or cell_type == 1:  # Empty or new cell
                otsl_seq.append(otsl_tag_maps["C"])
            elif cell_type == 2:  # Colspan
                otsl_seq.append(otsl_tag_maps["L"])
            elif cell_type == 3:  # Rowspan
                otsl_seq.append(otsl_tag_maps["U"])
            elif cell_type == 4:  # Both (not used in current logic)
                otsl_seq.append(otsl_tag_maps["X"])
        
        # Add newline tag at end of row
        otsl_seq.append(otsl_tag_maps["NL"])
    
    return otsl_seq, num_rows, num_cols


def process_fintabnet_sample(orig_data: Dict, sample_idx: int, split: str) -> Dict:
    """
    Process a single FinTabNet sample from original format to TFLOP format
    
    Args:
        orig_data: Original FinTabNet data
        sample_idx: Index of the sample (for filename)
        split: train/val/test
        
    Returns:
        Processed data in TFLOP format
    """
    
    # OTSL tag mapping (same as PubTabNet)
    otsl_tag_maps = {
        "C": "C-tag",
        "L": "L-tag",
        "U": "U-tag",
        "X": "X-tag",
        "NL": "NL-tag"
    }
    
    # Extract HTML tokens
    html_tokens = orig_data['html']['structure']['tokens']
    
    # Convert HTML tokens to string
    html_string = ''.join(html_tokens)
    
    # Generate OTSL sequence
    otsl_seq, num_rows, num_cols = convert_fintabnet_html_to_otsl(html_string, otsl_tag_maps)
    
    # Add structure tags (FinTabNet doesn't have thead/tbody distinction)
    # Treat all as tbody
    full_otsl = ['<thead>', '</thead>', '<tbody>'] + otsl_seq + ['</tbody>']
    
    # Process cells to create gold_coord
    cells = orig_data['html']['cells']
    gold_coord = []
    
    # Get table bbox to determine original image dimensions for normalization
    table_bbox = orig_data.get('bbox', [0, 0, 768, 768])
    
    # Find actual max dimensions from cell bboxes (more reliable than table bbox)
    max_x = 0
    max_y = 0
    for cell in cells:
        if 'bbox' in cell and cell['bbox']:
            bbox = cell['bbox']
            max_x = max(max_x, bbox[0], bbox[2])
            max_y = max(max_y, bbox[1], bbox[3])
    
    # Use the larger of table bbox or actual cell max for normalization
    orig_width = max(max_x, table_bbox[2]) if max_x > 0 else 768
    orig_height = max(max_y, table_bbox[3]) if max_y > 0 else 768
    
    # Calculate scale factors to normalize to 768x768
    scale_x = 768.0 / orig_width if orig_width > 768 else 1.0
    scale_y = 768.0 / orig_height if orig_height > 768 else 1.0
    
    for cell in cells:
        if 'bbox' in cell and cell['bbox']:
            # Filled cell: x1 y1 x2 y2 2 text
            bbox = cell['bbox']
            
            # Normalize coordinates to 768x768
            bbox = [
                min(767, bbox[0] * scale_x),
                min(767, bbox[1] * scale_y),
                min(767, bbox[2] * scale_x),
                min(767, bbox[3] * scale_y)
            ]
            tokens = cell.get('tokens', [])
            # Format text like PubTabNet (completely normal text)
            if tokens:
                # Join tokens and then remove extra spaces to create normal text
                text = ''.join(tokens).replace('  ', ' ').strip()
            else:
                text = ''
            
            # Format: "x1 y1 x2 y2 2 text"
            coord_str = f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} 2"
            if text:
                # Keep the formatted text as is (with spaces between chars)
                coord_str += f" {text}"
            gold_coord.append(coord_str)
        else:
            # Empty cell (same format as PubTabNet)
            gold_coord.append("-1.0 -1.0 -1.0 -1.0 1 ")
    
    # Create detection results (dr_coord) - same format as gold_coord for training
    # In real detection, this would come from a detection model
    dr_coord = {}
    for idx, coord_str in enumerate(gold_coord):
        # Parse bbox from coord string
        parts = coord_str.split(' ', 5)
        if not coord_str.startswith('-1.0'):
            bbox = [float(parts[i]) for i in range(4)]
            text = parts[5] if len(parts) > 5 else ''
            
            dr_coord[str(idx)] = [
                [bbox],  # List of bbox (could be multiple for merged cells)
                idx,     # Cell index
                text     # Cell text
            ]
    
    # Create filename
    if split == 'train':
        filename = f"fintabnet_train_{sample_idx:06d}.png"
    elif split == 'val':
        filename = f"fintabnet_val_{sample_idx:06d}.png"
    else:
        filename = f"fintabnet_test_{sample_idx:06d}.png"
    
    # Build final data structure
    processed = {
        'file_name': filename,
        'dr_coord': dr_coord,
        'gold_coord': gold_coord,
        'org_html': html_tokens,  # Keep original HTML tokens
        'otsl_seq': full_otsl,
        'num_rows': num_rows,
        'num_cols': num_cols,
        'split': split
    }
    
    return processed


def process_fintabnet_split(input_file: str, output_file: str, split: str, limit: int = None):
    """
    Process an entire FinTabNet split
    
    Args:
        input_file: Path to original FinTabNet JSONL file
        output_file: Path to output JSONL file
        split: train/val/test
        limit: Optional limit on number of samples to process
    """
    
    print(f"\nProcessing {split} split")
    print("="*60)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Statistics
    stats = {
        'total': 0,
        'processed': 0,
        'with_colspan': 0,
        'with_rowspan': 0,
        'errors': 0
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process samples
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for idx, line in enumerate(tqdm(infile, desc=f"Processing {split}")):
            if limit and idx >= limit:
                break
            
            stats['total'] += 1
            
            try:
                # Parse original data
                orig_data = json.loads(line.strip())
                
                # Process sample
                processed = process_fintabnet_sample(orig_data, idx, split)
                
                # Check for spans
                html_str = ''.join(processed['org_html'])
                if 'colspan' in html_str:
                    stats['with_colspan'] += 1
                if 'rowspan' in html_str:
                    stats['with_rowspan'] += 1
                
                # Write output
                outfile.write(json.dumps(processed) + '\n')
                stats['processed'] += 1
                
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                stats['errors'] += 1
    
    # Print statistics
    print(f"\n{split.upper()} Statistics:")
    print("-"*40)
    print(f"Total samples: {stats['total']:,}")
    print(f"Successfully processed: {stats['processed']:,}")
    print(f"Errors: {stats['errors']:,}")
    print(f"Tables with colspan: {stats['with_colspan']:,}")
    print(f"Tables with rowspan: {stats['with_rowspan']:,}")


def main():
    parser = argparse.ArgumentParser(description='Unified FinTabNet preprocessing for TFLOP')
    parser.add_argument('--input-dir', type=str,
                       default='/home/work/.nas/fintabnet/versions/1/fintabnet',
                       help='Directory containing original FinTabNet data')
    parser.add_argument('--output-dir', type=str,
                       default='/home/work/fintabnet_tflop/meta_data',
                       help='Output directory for processed data')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                       default='all', help='Which split to process')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Define splits to process
    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]
    
    print("\n" + "="*60)
    print("FinTabNet Unified Preprocessing for TFLOP")
    print("="*60)
    
    # Process each split
    for split in splits:
        # Define file paths
        if split == 'train':
            input_file = os.path.join(args.input_dir, 'FinTabNet_1.0.0_cell_train.jsonl')
            output_file = os.path.join(args.output_dir, 'dataset_train.jsonl')
        elif split == 'val':
            input_file = os.path.join(args.input_dir, 'FinTabNet_1.0.0_cell_val.jsonl')
            output_file = os.path.join(args.output_dir, 'dataset_val.jsonl')
        else:  # test
            input_file = os.path.join(args.input_dir, 'FinTabNet_1.0.0_cell_test.jsonl')
            output_file = os.path.join(args.output_dir, 'dataset_test.jsonl')
        
        # Check if input exists
        if not os.path.exists(input_file):
            print(f"\nWarning: {input_file} not found, skipping {split} split")
            continue
        
        # Process the split
        process_fintabnet_split(input_file, output_file, split, args.limit)
    
    print("\n" + "="*60)
    print("✅ FinTabNet preprocessing complete!")
    print("="*60)
    print(f"\nOutput files saved to: {args.output_dir}")
    print("\nNext steps:")
    print("1. Extract images from PDFs (if not already done)")
    print("2. Place images in corresponding directories:")
    print(f"   - {args.output_dir.replace('meta_data', 'images/train')}")
    print(f"   - {args.output_dir.replace('meta_data', 'images/val')}")
    print(f"   - {args.output_dir.replace('meta_data', 'images/test')}")
    print("3. Run training with the processed data")


if __name__ == "__main__":
    main()