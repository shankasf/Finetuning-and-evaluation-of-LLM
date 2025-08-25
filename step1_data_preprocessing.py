from datasets import load_dataset
from transformers import GPT2Tokenizer

# Load dataset (replace with your actual file paths)
dataset = load_dataset('text', data_files={'train': 'raw_data.txt', 'test': 'raw_data.txt'})

# Initialize the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token if not available
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save the tokenized dataset (for future use)
tokenized_datasets.save_to_disk('tokenized_data')
