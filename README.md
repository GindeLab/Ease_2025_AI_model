# Ease 2025 AI Model

This repository contains the implementation of an AI model for bug report analysis and quality assessment. The project includes tools for evaluating bug reports based on various quality metrics and generating summaries using advanced language models.

## Repository Structure

- `perfect_ctqrs.py`: Main Python script for bug report quality assessment
- `FIne_tune_code.ipynb`: Jupyter notebook for fine-tuning language models on bug report data
- `Generate_summary_llama.ipynb`: Notebook for generating summaries using LLaMA model
- `4oChatgpt_output.ipynb`: Notebook containing ChatGPT analysis outputs
- `filtered_bug_report.xlsx`: Dataset containing 3,966 well-structured bug reports

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (for model fine-tuning)
- Minimum 16GB RAM recommended
- At least 10GB free disk space

### Software Requirements
- Python 3.8 or higher
- NVIDIA CUDA Toolkit 12.6 or compatible version
- Git (for version control)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Ease_2025_AI_model.git
cd Ease_2025_AI_model
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Required Files

### Dataset Files
1. Main Dataset:
   - `filtered_bug_report.xlsx` (included in repository)
   - Contains 3,966 well-structured bug reports used for analysis

2. Additional Required Files (not included, must be created/obtained):
   - `final_unique_behaviours.csv`: Contains unique behavior patterns
   - `interface_element_words.csv`: List of UI element terms
   - `_negative_words.csv`: List of negative sentiment words

### Model Files
- Pre-trained model weights (not included in repository)
- Fine-tuned model checkpoints will be saved during training

## Usage Instructions

### 1. Data Preparation
1. Place the required CSV files in the project root directory:
   - `final_unique_behaviours.csv`
   - `interface_element_words.csv`
   - `_negative_words.csv`

2. Ensure `filtered_bug_report.xlsx` is present in the root directory

### 2. Quality Assessment
Run the quality assessment script:
```bash
python perfect_ctqrs.py
```

### 3. Model Fine-tuning
1. Open `FIne_tune_code.ipynb` in Jupyter Notebook/Lab
2. Follow the notebook cells sequentially
3. Adjust hyperparameters as needed in the configuration section

### 4. Generate Summaries
1. Open `Generate_summary_llama.ipynb`
2. Install Ollama if not already installed:
```bash
pip install ollama
```
3. Follow the notebook instructions for summary generation

## Troubleshooting

### Common Issues
1. Missing word list files:
   - Create the required CSV files with appropriate word lists
   - Use the fallback lists defined in the code if needed

2. CUDA/GPU issues:
   - Ensure NVIDIA drivers are up to date
   - Verify CUDA toolkit installation
   - Check GPU compatibility

3. Memory issues:
   - Reduce batch size in fine-tuning
   - Process data in smaller chunks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT

## Citation

If you use this code in your research, please cite:


