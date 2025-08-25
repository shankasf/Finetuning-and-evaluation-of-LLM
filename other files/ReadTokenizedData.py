from datasets import load_from_disk

# Load the dataset from the folder where it was saved
tokenized_datasets = load_from_disk('tokenized_data')

# Check the structure of the dataset
print(tokenized_datasets)

# Access the train split
train_dataset = tokenized_datasets['train']
print(train_dataset[0])  # This will print the first example in the dataset
