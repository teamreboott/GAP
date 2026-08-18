"""Summarise TEDS / TEDS-Struct / Position Accuracy for one checkpoint directory.

Reads the artefacts produced by test.py and evaluate_ted.py:
    <ckpt_dir>/ted_score_output.json          -> TEDS, TEDS-Struct
    <ckpt_dir>/full_model_inference_0_1.json  -> Position Accuracy

Usage:
    python3 scripts/report_metrics.py <ckpt_dir>
"""

import argparse
import json
import os

from bs4 import BeautifulSoup


def parse_html_table(html_string):
    """Flatten an HTML table into (row, col) -> cell-content, honouring spans."""
    html_string = html_string.replace("<html><body><table>", "").replace(
        "</table></body></html>", ""
    )
    soup = BeautifulSoup(f"<table>{html_string}</table>", "html.parser")

    cells, occupied, row_idx = [], set(), 0
    for section in ("thead", "tbody"):
        section_elem = soup.find(section)
        if section_elem is None:
            continue
        for tr in section_elem.find_all("tr", recursive=False):
            col_idx = 0
            for td in tr.find_all(["td", "th"], recursive=False):
                while (row_idx, col_idx) in occupied:
                    col_idx += 1
                rowspan = int(td.get("rowspan", 1))
                colspan = int(td.get("colspan", 1))
                cells.append(
                    {"row": row_idx, "col": col_idx, "content": td.get_text(strip=True)}
                )
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if (r, c) != (row_idx, col_idx):
                            occupied.add((r, c))
                col_idx += colspan
            row_idx += 1
    return cells


def mean(values):
    return sum(values) / len(values) if values else 0.0


def teds_scores(ckpt_dir):
    """TEDS and TEDS-Struct, averaged over the test set."""
    path = os.path.join(ckpt_dir, "ted_score_output.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    # each row, per evaluate_ted.py:
    #   [file_name, pred_html, gold_html, edit_distance, teds_struct, teds]
    return (
        mean([r[5] for r in data if len(r) > 5]),  # TEDS (content + structure)
        mean([r[4] for r in data if len(r) > 4]),  # TEDS-Struct (structure only)
    )


def position_accuracy(ckpt_dir):
    """Fraction of ground-truth cells whose content lands at the correct grid slot."""
    path = os.path.join(ckpt_dir, "full_model_inference_0_1.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)

    correct = total = 0
    for sample in data.values():
        gt_cells = parse_html_table(sample["answer_string"])
        pred_map = {
            (c["row"], c["col"]): c["content"]
            for c in parse_html_table(sample["pred_string"])
        }
        for cell in gt_cells:
            if pred_map.get((cell["row"], cell["col"])) == cell["content"]:
                correct += 1
        total += len(gt_cells)
    return 100.0 * correct / total if total else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt_dir", help="checkpoint directory holding the artefacts")
    args = parser.parse_args()

    teds, teds_s = teds_scores(args.ckpt_dir)
    pa = position_accuracy(args.ckpt_dir)

    def fmt(x):
        return f"{x * 100:.2f}" if x is not None else "n/a"

    print(f"checkpoint : {args.ckpt_dir}")
    print(f"TEDS       : {fmt(teds)}")
    print(f"TEDS-Struct: {fmt(teds_s)}")
    print(f"PA         : {pa:.2f}" if pa is not None else "PA         : n/a")


if __name__ == "__main__":
    main()
