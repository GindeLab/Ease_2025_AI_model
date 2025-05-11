import pandas as pd

def extract_unique_behaviours(input_excel, output_csv):
    # Load the Excel file
    df = pd.read_excel(input_excel)
    
    # Extract the 'user_behviou' column and split by comma
    unique_behaviours = set()
    
    for behaviours in df['Unique_Behaviours'].dropna():
        unique_behaviours.update(behaviours.split(','))
    
    # Convert to DataFrame
    unique_behaviours_df = pd.DataFrame({'Unique_Behaviours': list(unique_behaviours)})
    
    # Save to CSV
    unique_behaviours_df.to_csv(output_csv, index=False)
    print(f"Unique behaviours saved to {output_csv}")

# Example usage
extract_unique_behaviours("../Final_12k_3000_Ollama_user_behviour.xlsx", "unique_behaviours.csv")