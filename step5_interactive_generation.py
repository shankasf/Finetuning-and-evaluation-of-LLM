import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_gpt2')

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Resize the model embeddings if new tokens were added
model.resize_token_embeddings(len(tokenizer))

# Move model to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Set model to evaluation mode
model.eval()

# Interactive text generation
while True:
    prompt = input("Enter your text prompt (or type 'exit' to quit): ")
    if prompt.lower() == 'exit':
        break
    
    # Tokenize the prompt with padding and return the attention mask
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Move inputs to the correct device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Generate text based on the prompt
    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=150, pad_token_id=tokenizer.pad_token_id)
    
    # Decode and print the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}\n")
