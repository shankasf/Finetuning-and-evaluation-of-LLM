import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_from_disk

# Load the fine-tuned model and tokenizer
model_name_or_path = "fine_tuned_gpt2"  # Replace with your fine-tuned model path
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Assign EOS token as the padding token for GPT-2
tokenizer.pad_token = tokenizer.eos_token

# Load the model and update the pad_token_id
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
model.config.pad_token_id = tokenizer.pad_token_id

# Load the tokenized dataset from the 'tokenized_data' folder and select the 'test' split
dataset = load_from_disk("tokenized_data")  # Load from your tokenized data folder
test_dataset = dataset['test']  # Select the 'test' split

# Ensure proper padding and truncation during tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize the dataset
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Add labels to the dataset (adapt the logic based on your task)
def add_labels(example):
    # Replace with your logic to assign labels
    label_mapping = {"class_a": 0, "class_b": 1}  # Example label mapping
    example["labels"] = label_mapping.get(example["text"], 0)  # Assign a default or calculated label
    return example

# Apply the label adding function
tokenized_test_dataset = tokenized_test_dataset.map(add_labels)

# Convert the tokenized features into PyTorch tensors
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create the DataLoader with a batch size of 1 to bypass padding issues
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=1)

# Set the model to evaluation mode
model.eval()

# Initialize evaluation metrics
total_loss = 0.0
num_batches = 0

# Iterate through the test data
for batch in test_dataloader:
    input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    labels = batch['labels'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

    total_loss += loss.item()
    num_batches += 1

# Calculate the average loss
average_loss = total_loss / num_batches
print(f"Average evaluation loss: {average_loss}")
