#!/usr/bin/env python3
"""
Convert FinTabNet HTML to OTSL sequence - Version 2
Properly handles colspan and rowspan
"""

import re
from typing import List, Tuple, Dict
import numpy as np

def convert_fintabnet_to_otsl(html_string: str, otsl_tag_maps: Dict[str, str]) -> Tuple[List[str], int, int]:
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

def test_conversion():
    """Test with sample HTML"""
    
    test_cases = [
        # Simple 2x2 table
        ('<table><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>', 
         "Simple 2x2"),
        
        # Table with colspan
        ('<table><tr><td colspan="2"></td></tr><tr><td></td><td></td></tr></table>',
         "2x2 with colspan=2 in first row"),
        
        # Table with rowspan
        ('<table><tr><td rowspan="2"></td><td></td></tr><tr><td></td></tr></table>',
         "2x2 with rowspan=2 in first cell"),
        
        # Complex table
        ('<table><tr><td rowspan="2" colspan="2"></td><td></td></tr><tr><td></td></tr></table>',
         "2x3 with rowspan=2 colspan=2"),
    ]
    
    otsl_maps = {
        "C": "C-tag",
        "L": "L-tag",
        "U": "U-tag",
        "X": "X-tag",
        "NL": "NL-tag"
    }
    
    for html, desc in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {desc}")
        print(f"HTML: {html}")
        
        otsl, rows, cols = convert_fintabnet_to_otsl(html, otsl_maps)
        print(f"Grid: {rows}x{cols}")
        print(f"OTSL sequence:")
        
        # Format output nicely
        idx = 0
        for r in range(rows):
            row_tags = []
            for c in range(cols):
                if idx < len(otsl) and otsl[idx] != 'NL-tag':
                    row_tags.append(otsl[idx])
                    idx += 1
            if idx < len(otsl) and otsl[idx] == 'NL-tag':
                idx += 1
            print(f"  Row {r}: {' '.join(row_tags)}")

if __name__ == "__main__":
    test_conversion()