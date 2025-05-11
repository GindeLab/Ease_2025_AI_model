import pandas as pd
import re

# Load your Excel file
df = pd.read_excel('Unique_Detailed_Bug_Reports_top100.xlsx')

# Function to extract the three sections from text
def extract_sections(text):
    # Handle non-string types
    if not isinstance(text, str):
        return '', '', ''
    
    # Initialize default values
    s2r = ''
    actual = ''
    expected = ''
    
    # Look for "Steps to reproduce" section
    s2r_match = re.search(r'Steps to reproduce:[\s\n]*(.*?)(?:\n\s*Actual results:|\Z)', text, re.DOTALL | re.IGNORECASE)
    if s2r_match:
        s2r = s2r_match.group(1).strip()
    
    # Look for "Actual results" section
    actual_match = re.search(r'Actual results:[\s\n]*(.*?)(?:\n\s*Expected results:|\Z)', text, re.DOTALL | re.IGNORECASE)
    if actual_match:
        actual = actual_match.group(1).strip()
    
    # Look for "Expected results" section - improved pattern
    expected_match = re.search(r'Expected results:[\s\n]*(.*?)(?:\n\s*\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if expected_match:
        expected = expected_match.group(1).strip()
    
    return s2r, actual, expected

# Check what column contains the text data you need to process
print("Available columns:", df.columns.tolist())

# Let's determine the actual text column name
text_column = None
for col in df.columns:
    # Try to find a column that contains text with our target sections
    sample_values = df[col].dropna().astype(str).head(5).tolist()
    for val in sample_values:
        if 'Steps to reproduce:' in val or 'Actual results:' in val:
            text_column = col
            break
    if text_column:
        break

if not text_column:
    print("Could not automatically identify the text column. Please specify the column name.")
    # Ask for column name or try to use a common column name
    potential_cols = ['description', 'text', 'content', 'raw_text', 'comments', 'bug_description']
    for col in potential_cols:
        if col in df.columns:
            text_column = col
            break
    
    # If still not found, default to the first text-like column
    if not text_column:
        for col in df.columns:
            if df[col].dtype == object:
                text_column = col
                break
        # Last resort: use the first column
        if not text_column and len(df.columns) > 0:
            text_column = df.columns[0]

print(f"Using column '{text_column}' for text extraction")

# Apply the extraction function to create new columns
results = []
for i, row in df.iterrows():
    try:
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        results.append(extract_sections(text))
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        results.append(('', '', ''))

# Convert results to DataFrame and assign to new columns
result_df = pd.DataFrame(results, columns=['Steps_to_Reproduce', 'Actual_Results', 'Expected_Results'])
df = pd.concat([df, result_df], axis=1)

# Print some samples to verify extraction worked
print("\nSample Extractions (first 3 rows):")
for i in range(min(3, len(df))):
    print(f"\nRow {i}:")
    print(f"Steps to Reproduce: {df['Steps_to_Reproduce'].iloc[i][:100]}...")
    print(f"Actual Results: {df['Actual_Results'].iloc[i][:100]}...")
    print(f"Expected Results: {df['Expected_Results'].iloc[i][:100]}...")

# Create masked versions
# Version 1: Mask Steps to Reproduce
df['Masked_S2R_Version'] = df.apply(
    lambda row: f"Steps to reproduce:\n[MASKED]\n\nActual results:\n{row['Actual_Results']}\n\nExpected results:\n{row['Expected_Results']}", 
    axis=1
)

# Version 2: Mask Actual Results
df['Masked_Actual_Version'] = df.apply(
    lambda row: f"Steps to reproduce:\n{row['Steps_to_Reproduce']}\n\nActual results:\n[MASKED]\n\nExpected results:\n{row['Expected_Results']}", 
    axis=1
)
# Version 2: Mask Actual Results
df['Masked_Expected_Version'] = df.apply(
    lambda row: f"Steps to reproduce:\n{row['Steps_to_Reproduce']}\n\nActual results:\n{row['Expected_Results']}\n\nExpected results:\n[MASKED]", 
    axis=1
)

# Save the processed data to a new Excel file
df.to_excel('Processed_Bug_Reports.xlsx', index=False)
print("\nProcessing complete. Results saved to 'Processed_Bug_Reports.xlsx'")

# Optional: Create a separate file with just the extracted and masked columns
extracted_df = df[['Steps_to_Reproduce', 'Actual_Results', 'Expected_Results', 
                 'Masked_S2R_Version', 'Masked_Actual_Version']]
extracted_df.to_excel('Extracted_Bug_Reports.xlsx', index=False)
print("Extracted columns saved to 'Extracted_Bug_Reports.xlsx'")