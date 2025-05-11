import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download VADER lexicon (run this once)
nltk.download('vader_lexicon')

# Load the CSV file
file_path = "unique_user_behaviour_words.csv"
df = pd.read_csv(file_path)
print(df)
# Correct column name (replace with actual column name from CSV)
column_name = "Unique Words"  # Ensure this matches the actual column name in your file
df[column_name] = df[column_name].astype(str)  # Ensure column is string

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to clean words by removing special characters
def clean_word(word):
    return re.sub(r'[^a-zA-Z0-9]', '', word)  # Remove special characters

# Function to determine sentiment of a word
def get_sentiment_vader(word):
    return sia.polarity_scores(word)['compound']

# Extracting and cleaning words
filtered_words_vader = set()
for entry in df[column_name].dropna():  # Ensure no NaN values
    words = entry.split()  # Split by spaces
    for word in words:
        cleaned_word = clean_word(word.lower())  # Clean and convert to lowercase
        if cleaned_word:  # Check if the word is valid
            sentiment_score = get_sentiment_vader(cleaned_word)  # Get sentiment score
            if sentiment_score < 0:  # Check if it's negative
                filtered_words_vader.add(cleaned_word)

# Convert to a DataFrame
sentiment_filtered_words_df_vader = pd.DataFrame(sorted(filtered_words_vader), columns=["Negative Words"])

# Save to CSV
negative_csv_path = "sentiment_filtered_negative_words_vader.csv"
sentiment_filtered_words_df_vader.to_csv(negative_csv_path, index=False)

print(f"Negative words saved to {negative_csv_path}")
