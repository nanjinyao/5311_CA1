import json
import re

def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def extract_chinese_lines(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        for cell_idx, cell in enumerate(nb['cells']):
            source = cell.get('source', [])
            # source is a list of strings
            # But sometimes it's a single string? No, standard NB format is list of strings.
            if isinstance(source, str):
                source = [source]
                
            for line_idx, line in enumerate(source):
                if contains_chinese(line):
                    # Write: CellIdx|LineIdx|Content
                    # Use a unique separator
                    out.write(f"{cell_idx}|{line_idx}|{line}")

if __name__ == "__main__":
    extract_chinese_lines('ad_v7.ipynb', 'chinese_lines.txt')
