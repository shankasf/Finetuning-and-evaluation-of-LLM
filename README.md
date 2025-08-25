# LLM Fine-tuning and Evaluation Project

This repository provides a full workflow for fine-tuning, evaluating, and interacting with Large Language Models (LLMs). It is structured into modular steps, each handling one part of the process from data preprocessing to interactive generation.

---

## Project Structure

```
.
├── fine_tuned_gpt2/           # Directory containing the fine-tuned model
├── loaded_model/              # Directory for loading pretrained or fine-tuned models
├── other files/               # Miscellaneous supporting files
├── tokenized_data/            # Tokenized training data for model input
├── raw_data.txt               # Original dataset for training/evaluation
├── step1_data_preprocessing.py
├── step2_model_loading.py
├── step3_fine_tuning.py
├── step4_evaluation.py
├── step5_interactive_generation.py
└── README.md
```

---

## Workflow

### 1. Data Preprocessing (`step1_data_preprocessing.py`)

* Loads raw text data (`raw_data.txt`).
* Cleans and tokenizes the data.
* Saves processed datasets in `tokenized_data/` for training.

### 2. Model Loading (`step2_model_loading.py`)

* Loads a pretrained base model (e.g., GPT-2).
* Prepares the model for fine-tuning.
* Supports configuration for GPU/CPU environments.

### 3. Fine-Tuning (`step3_fine_tuning.py`)

* Fine-tunes the base model on domain-specific data.
* Saves checkpoints and the final fine-tuned model in `fine_tuned_gpt2/`.

### 4. Evaluation (`step4_evaluation.py`)

* Evaluates the fine-tuned model using test data.
* Computes metrics such as loss, perplexity, and accuracy.
* Logs evaluation results for comparison.

### 5. Interactive Generation (`step5_interactive_generation.py`)

* Loads the fine-tuned model.
* Provides a command-line interface for generating text.
* Allows experimentation with prompts and generation parameters.

---

## Getting Started

### Requirements

* Python 3.8+
* PyTorch
* Hugging Face Transformers
* Datasets
* Tokenizers

Install dependencies:

```bash
pip install torch transformers datasets
```

### Running the Pipeline

1. Prepare raw text data (`raw_data.txt`).
2. Run preprocessing:

   ```bash
   python step1_data_preprocessing.py
   ```
3. Load and prepare the base model:

   ```bash
   python step2_model_loading.py
   ```
4. Fine-tune the model:

   ```bash
   python step3_fine_tuning.py
   ```
5. Evaluate the fine-tuned model:

   ```bash
   python step4_evaluation.py
   ```
6. Start interactive text generation:

   ```bash
   python step5_interactive_generation.py
   ```

---

## Notes

* The project is modular: you can run each step independently.
* Use `loaded_model/` to resume from a pretrained checkpoint.
* Ensure GPU availability for fine-tuning large models.
* Training data should be domain-relevant for better performance.

---

## Next Steps

* Add support for multiple model architectures (e.g., GPT-Neo, LLaMA).
* Extend evaluation with BLEU/ROUGE scores for text quality.
* Provide a Streamlit-based web interface for interactive generation.
