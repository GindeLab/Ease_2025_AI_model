import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_unique_behaviours(input_excel, output_csv, chunk_size=1000, similarity_threshold=0.85):
    # Load the Excel file in chunks
    chunks = pd.read_csv(input_excel, chunksize=chunk_size)
    unique_behaviours = set()
    
    for chunk in chunks:
        for behaviours in chunk['Unique_Behaviours'].dropna():
            unique_behaviours.update(behaviours.split(','))
    
    # Convert to list
    unique_behaviours = list(unique_behaviours)
    
    # Compute TF-IDF vectors for similarity comparison
    vectorizer = TfidfVectorizer().fit_transform(unique_behaviours)
    vectors = vectorizer.toarray()
    similarity_matrix = cosine_similarity(vectors)
    
    # Remove similar words
    def remove_similar_words(words, similarity_matrix, threshold):
        unique_words = []
        removed_indices = set()
        
        for i in range(len(words)):
            if i in removed_indices:
                continue
            unique_words.append(words[i])
            for j in range(i + 1, len(words)):
                if similarity_matrix[i, j] > threshold:
                    removed_indices.add(j)
        
        return unique_words
    
    filtered_behaviours = remove_similar_words(unique_behaviours, similarity_matrix, similarity_threshold)
    
    # Save cleaned unique behaviors to a new CSV
    cleaned_df = pd.DataFrame({'Filtered_Unique_Behaviours': filtered_behaviours})
    cleaned_df.to_csv(output_csv, index=False)
    print(f"Unique behaviours saved to {output_csv}")

# Example usage
extract_unique_behaviours("unique_behaviours.csv", "cleaned_unique_behaviours.csv")
