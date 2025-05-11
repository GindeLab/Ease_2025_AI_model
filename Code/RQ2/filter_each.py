import pandas as pd
import re

# Function to extract specific sections from raw text safely
def extract_section(text, section_name):
    if not isinstance(text, str):
        return ""  # Ensure we only process strings

    pattern = rf"\*{section_name}\*\n(.*?)\n\n"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

# Load the Excel file
input_file = "CTQRS_filtered_bug_report_scores.xlsx"

try:
    df = pd.read_excel(input_file)
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit(1)

# Ensure there's a column named 'raw_text'
if 'raw_text' not in df.columns:
    print("Error: The Excel file must contain a 'raw_text' column.")
    exit(1)

# Process each bug report safely
filtered_reports = []
for index, row in df.iterrows():
    raw_text = row.get("raw_text", "")

    if pd.isna(raw_text):  # Handle NaN values
        raw_text = ""

    if not isinstance(raw_text, str):
        print(f"Skipping row {index}: raw_text is not a string")
        continue

    try:
        steps_to_reproduce = extract_section(raw_text, "Steps to reproduce")
        actual_result = extract_section(raw_text, "Actual result") or extract_section(raw_text, "Actual results")
        expected_result = extract_section(raw_text, "Expected result") or extract_section(raw_text, "Expected results")

        # Check if any section has more than 50 characters
        if (
            len(steps_to_reproduce) > 100 or
            len(actual_result) > 50 or
            len(expected_result) > 50
        ):
            filtered_reports.append(row)

    except Exception as e:
        print(f"Error processing row {index}: {e}")

# Create a new DataFrame with filtered reports
filtered_df = pd.DataFrame(filtered_reports)

# Save to a new Excel file with error handling
output_file = "CTQRS_filtered_bug_report_scores_filtered.xlsx"

try:
    filtered_df.to_excel(output_file, index=False)
    print(f"Filtered bug reports saved to {output_file}")
except Exception as e:
    print(f"Error saving filtered Excel file: {e}")
