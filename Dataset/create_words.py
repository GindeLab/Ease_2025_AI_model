import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the CSV file
file_path = "unique_words.csv"
df = pd.read_csv(file_path)
print(df)
# Correct column name (replace with actual column name from CSV)
column_name = "word "  # Ensure this matches the actual column name in your file
df[column_name] = df[column_name].astype(str)  # Ensure column is string

# Function to clean words by removing special characters
def clean_word(word):
    return re.sub(r'[^a-zA-Z0-9]', '', word)  # Remove special characters

# Function to extract nouns from a given list of words
def extract_nouns(word_list):
    words = [clean_word(word) for word in word_list if clean_word(word)]
    tagged_words = pos_tag(words)  # Perform POS tagging
    nouns = {word for word, tag in tagged_words if tag.startswith('NN')}  # Extract nouns
    return nouns

# Extracting and cleaning words
all_nouns = set()
for entry in df[column_name].dropna():  # Ensure no NaN values
    words = word_tokenize(entry)  # Tokenize the entry
    nouns = extract_nouns(words)  # Extract nouns
    all_nouns.update(nouns)  # Add to the set

# Convert to a DataFrame
nouns_df = pd.DataFrame(sorted(all_nouns), columns=["Nouns"])

# Save to CSV
nouns_csv_path = "extracted_nouns.csv"
nouns_df.to_csv(nouns_csv_path, index=False)

print(f"Nouns saved to {nouns_csv_path}")
