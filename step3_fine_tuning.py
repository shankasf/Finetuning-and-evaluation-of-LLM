import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_dataset

# Load raw text dataset
dataset = load_dataset('text', data_files={'train': 'raw_data.txt'})

# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Add padding token as the eos token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Ensure that the dataset contains 'input_ids' and 'attention_mask'
print(tokenized_datasets['train'][0])

# Use a data collator that dynamically pads the inputs
collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

# Create DataLoader
train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=8, shuffle=True, collate_fn=collator)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Resize model embeddings if new tokens were added to the tokenizer
model.resize_token_embeddings(len(tokenizer))

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tuning loop
epochs = 3
for epoch in range(epochs):
    for batch in train_dataloader:
        # Move batch to the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Save the fine-tuned model and tokenizer
model.save_pretrained('fine_tuned_gpt2')
tokenizer.save_pretrained('fine_tuned_gpt2')

