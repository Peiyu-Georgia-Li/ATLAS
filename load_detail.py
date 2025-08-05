from datasets import load_dataset
model_name= "Qwen/Qwen2.5-72B-Instruct"
dataset_path = f"open-llm-leaderboard/{model_name.replace('/', '__')}-details"


data = load_dataset(
    dataset_path,
    name=f"{model_name.replace('/', '__')}__leaderboard_mmlu_pro",
    split="latest"
)
print(data)
print(data[0])  # Print the first example
# Or to see just specific fields:
print(f"Prompt: {data[0]['doc']}")
print(f"Target answer: {data[0]['target']}")
print(f"Model response: {data[0]['resps']}")

