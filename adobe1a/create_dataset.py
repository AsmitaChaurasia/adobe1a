import fitz
import pandas as pd
import json
import os
import re

# --- Helper Functions for Feature Extraction (Logic Unchanged) ---

def get_numbering_pattern(text: str) -> int:
    """Identifies numbering patterns like '1.' or '1.1.' at the start of a line."""
    if len(text.split()) > 15: return 0
    if re.match(r'^\d+\.\s', text): return 1
    if re.match(r'^\d+\.\d+(\.\d+)*\s', text): return 2
    return 0

def is_line_centered(line_bbox, page_width, tolerance=40):
    """Checks if a line of text is horizontally centered on the page."""
    line_midpoint = (line_bbox[0] + line_bbox[2]) / 2
    page_midpoint = page_width / 2
    return abs(line_midpoint - page_midpoint) < tolerance

def get_caps_ratio(text: str) -> float:
    """Calculates the ratio of uppercase characters in a string."""
    if not text or not any(c.isalpha() for c in text): return 0
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars: return 0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

def is_toc_page(page: fitz.Page) -> bool:
    """Heuristically detects if a page is a Table of Contents."""
    lines = page.get_text("text").strip().split('\n')
    if len(lines) < 5: return False
    lines_ending_with_number = sum(1 for line in lines if line.strip() and line.strip().split()[-1].isdigit())
    return (lines_ending_with_number / len(lines)) > 0.5

# --- Main Script (Logic Unchanged) ---

def create_gold_standard_dataset(pdf_dir: str, json_dir: str, output_csv_path: str):
    """
    Processes PDFs and their corresponding ground truth JSON files to create a feature-rich
    dataset for training a heading classification model.
    """
    all_dataframes = []
    pdf_filenames = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    print(f"Found {len(pdf_filenames)} PDFs in '{pdf_dir}'")

    # The entire PDF processing loop remains identical
    for pdf_filename in pdf_filenames:
        pdf_path = os.path.join(pdf_dir, pdf_filename)
        json_filename = os.path.splitext(pdf_filename)[0] + '.json'
        json_path = os.path.join(json_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"Warning: No matching JSON for {pdf_filename}. Skipping.")
            continue
        
        print(f"\nProcessing: {pdf_filename}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        
        true_headings = {item['text'].strip(): item['level'] for item in ground_truth.get('outline', [])}
        true_title = ground_truth.get('title', '').strip()
        
        doc = fitz.open(pdf_path)
        features_list = []
        for page_num, page in enumerate(doc):
            if is_toc_page(page):
                print(f"  -> Skipping page {page_num + 1} (detected as Table of Contents).")
                continue

            prev_block_y1 = 0
            page_height = page.rect.height
            page_width = page.rect.width
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block['type'] == 0:
                    space_above = block['bbox'][1] - prev_block_y1 if prev_block_y1 > 0 else 0
                    for line in block.get("lines", []):
                        if not line.get('spans'): continue
                        
                        first_span = line['spans'][0]
                        full_line_text = "".join(s['text'] for s in line['spans']).strip()
                        if not full_line_text: continue

                        features = {
                            "line_size": first_span['size'],
                            "is_bold": 1 if "bold" in first_span['font'].lower() else 0,
                            "word_count": len(full_line_text.split()),
                            "numbering_pattern": get_numbering_pattern(full_line_text),
                            "is_centered": 1 if is_line_centered(line['bbox'], page_width) else 0,
                            "space_above": space_above,
                            "ratio_of_caps": get_caps_ratio(full_line_text),
                            "is_on_first_page": 1 if page_num == 0 else 0,
                            "vertical_position": line['bbox'][1] / page_height
                        }
                        
                        if full_line_text == true_title:
                            features['label'] = "Title"
                        else:
                            features['label'] = true_headings.get(full_line_text, "Body Text")
                        
                        features_list.append(features)
                    
                    prev_block_y1 = block['bbox'][3]
        doc.close()
        all_dataframes.append(pd.DataFrame(features_list))
    
    if all_dataframes:
        final_dataset = pd.concat(all_dataframes, ignore_index=True)
        final_dataset.to_csv(output_csv_path, index=False)
        print(f"\n--- Gold Standard Dataset Generated ---")
        print(f"Saved a total of {len(final_dataset)} lines to {output_csv_path}")
    else:
        print("\nNo dataframes were created. The output file was not generated.")


# --- Main execution block ---
if __name__ == "__main__":
    # Define paths relative to the script's location.
    # This assumes the script is inside the 'adobe1a' folder.
    pdf_directory = "training_data/pdfs"
    json_directory = "training_data/ground_truth_jsons"
    
    # Define the output path and create the directory if it doesn't exist
    output_dir = "trained_model"
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, "gold_standard_dataset.csv")

    # Check if the input directories exist before running
    if not os.path.isdir(pdf_directory) or not os.path.isdir(json_directory):
        print(f"Error: Could not find input directories.")
        print(f"Please make sure you are running this script from inside the 'adobe1a' folder.")
    else:
        # Run the main function with the defined paths
        create_gold_standard_dataset(pdf_directory, json_directory, output_csv_path)