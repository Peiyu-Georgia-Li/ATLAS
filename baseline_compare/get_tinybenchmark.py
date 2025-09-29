from datasets import load_dataset
import pandas as pd
import csv
import re

# Load the TruthfulQA dataset
DATASET_NAME = 'TruthfulQA'
tiny_data = load_dataset('tinyBenchmarks/tiny'+DATASET_NAME, 'multiple_choice')['validation']

# DATASET_NAME = 'GSM8K'
# tiny_data = load_dataset('tinyBenchmarks/tinyGSM8K', 'main')['test']

# DATASET_NAME = 'ARC'
# tiny_data = load_dataset('tinyBenchmarks/tinyAI2_arc', 'ARC-Challenge')['test']

# DATASET_NAME = 'Hellaswag'
# tiny_data = load_dataset('tinyBenchmarks/tinyHellaswag')['validation']


# DATASET_NAME = 'Winogrande'
# tiny_data = load_dataset('tinyBenchmarks/tinyWinogrande', 'winogrande_xl')['validation']



# Path to the CSV file with item IDs and prompts
ID_PATH = '/store01/nchawla/pli9/llmbenchmark/metabench_data/benchmark-data/'+DATASET_NAME.lower()+'_prompts.csv'

# Read the CSV file with item IDs
prompts_df = pd.read_csv(ID_PATH)

# We'll directly check if questions are in prompts

# Create a list to store the matched item IDs
item_ids = []
questions = []

if 'question' in tiny_data:
    tiny_data_questions = tiny_data['question']
else:
    tiny_data_questions = tiny_data['input_formatted']

# Match each question in tiny_data to an item ID
for question in tiny_data_questions:
    questions.append(question)
    found = False
    for _, row in prompts_df.iterrows():
        prompt = row['prompt']
        # Check if the question is in the prompt
        if question in prompt:
            item_ids.append(row['item'])
            found = True
            break
    if not found:
        item_ids.append('not_found')

# Create a DataFrame with questions and item IDs
results_df = pd.DataFrame({
    'item_id': item_ids,
    'question': questions
})

# Save the results to a CSV file
output_path = f'tiny{DATASET_NAME}_item_ids.csv'
results_df.to_csv(output_path, index=False)

print(f"Saved {len(item_ids)} questions with item IDs to {output_path}")
print(f"Sample of the data:\n{results_df.head()}")
