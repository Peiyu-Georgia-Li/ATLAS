"""
Script to filter models from the open-llm-leaderboard based on their scores for a specific task.
Saves models with acc_norm,none > 0.15 to a JSON file.
"""
import requests
import re
import ast
import json
import time
import os
import pickle
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Default constants
TASK_NAME = "leaderboard_bbh_tracking_shuffled_objects_five_objects"
SCORE_THRESHOLD = 0.15
OUTPUT_FILE = "models_above_threshold_2.json"
REQUEST_DELAY = 0.2  # Seconds to delay between requests
MAX_RETRIES = 3      # Maximum number of retries for rate limited requests
MAX_WORKERS = 3      # Maximum number of concurrent threads
CACHE_DIR = "readme_cache"
RESUME_FILE = "resume_state.json"
POPULAR_MODELS_FILE = "popular_models.txt"

# Create cache directory
os.makedirs(CACHE_DIR, exist_ok=True)

def get_popular_models():
    """Get a list of popular models to prioritize."""
    popular_models = [
        "Qwen/Qwen2.5-72B-Instruct",
        "meta-llama/Llama-3-70b-instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-7b",
        "google/gemma-2-9b-it",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "microsoft/phi-2",
        "microsoft/phi-3-mini",
        "microsoft/phi-3-small",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-3-haiku-20240307",
        "01-ai/Yi-34B",
        "01-ai/Yi-6B",
        "openai/gpt-4",
        "openai/gpt-3.5-turbo",
        "cohere/command-r-plus",
        "cohere/command-r",
        "stabilityai/stablelm-2-zephyr-1_6b",
    ]
    
    # Save popular models to file if it doesn't exist
    if not os.path.exists(POPULAR_MODELS_FILE):
        with open(POPULAR_MODELS_FILE, 'w') as f:
            for model in popular_models:
                f.write(f"{model}\n")
    else:
        # Read from file if it exists (allows user customization)
        with open(POPULAR_MODELS_FILE, 'r') as f:
            popular_models = [line.strip() for line in f if line.strip()]
    
    return popular_models

def get_model_list(prioritize_popular=True):
    """Get a list of models from the local open_llm_leaderboard_data.json file."""
    try:
        # Read from local JSON file
        with open('open_llm_leaderboard_data.json', 'r') as file:
            leaderboard_data = json.load(file)
        
        models = []
        
        # Extract models from the leaderboard data
        for entry in leaderboard_data:
            if 'model' in entry and 'name' in entry['model']:
                model_name = entry['model']['name']
                if model_name:  # Make sure we have a valid name
                    models.append(model_name)
        
        print(f"Found {len(models)} models in the local leaderboard data.")
        
        if prioritize_popular:
            # Move popular models to the front of the list
            popular_models = get_popular_models()
            # Create a new list with popular models first
            prioritized_models = []
            for model in popular_models:
                if model in models:
                    prioritized_models.append(model)
                    models.remove(model)
            
            # Add the remaining models
            prioritized_models.extend(models)
            
            print(f"Prioritized {len(popular_models)} popular models.")
            return prioritized_models
        else:
            return models
    except Exception as e:
        print(f"Error getting model list from local file: {e}")
        # Return popular models as fallback
        return get_popular_models()

def get_model_score(model_name):
    """Get the score for a specific model and task."""
    # Check cache first
    cache_file = f"readme_cache/{model_name.replace('/', '_').replace(':', '_')}.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
                print(f"{model_name}: Score = {cached_result['score']} (cached)")
                return cached_result
        except Exception as e:
            print(f"Error loading cache for {model_name}: {e}")
            # Continue if cache loading fails
    
    # Format model ID for URL
    model_id = model_name.replace('/', '__')
    readme_url = f"https://huggingface.co/datasets/open-llm-leaderboard/{model_id}-details/resolve/main/README.md"
    
    # Implement retry logic with exponential backoff
    for retry in range(MAX_RETRIES + 1):
        try:
            # Add delay between requests to avoid rate limiting
            if retry > 0:
                # Exponential backoff
                wait_time = (2 ** (retry - 1)) * REQUEST_DELAY * 3
                print(f"Rate limited for {model_name}, waiting {wait_time:.2f}s before retry {retry}/{MAX_RETRIES}")
                time.sleep(wait_time)
            else:
                # Regular delay for first attempt
                time.sleep(REQUEST_DELAY)
            
            response = requests.get(readme_url)
            
            if response.status_code == 429:
                if retry == MAX_RETRIES:
                    print(f"Failed to fetch README for {model_name} after {MAX_RETRIES} retries: HTTP 429")
                    return None
                continue  # Try again with longer delay
            
            if response.status_code != 200:
                print(f"Failed to fetch README for {model_name}: HTTP {response.status_code}")
                return None
            
            readme_content = response.text
            
            # Try to find the "Latest results" section with the URL and Python dictionary data
            results_pattern = r'## Latest results[\s\S]*?\(.*?\):[\s\S]*?```python\s*({[\s\S]*?})\s*```'
            results_match = re.search(results_pattern, readme_content)
            
            if not results_match:
                # Try alternative pattern without python tag
                results_pattern = r'## Latest results[\s\S]*?\(.*?\):[\s\S]*?```\s*({[\s\S]*?})\s*```'
                results_match = re.search(results_pattern, readme_content)
                
            if not results_match:
                print(f"No results found for {model_name}")
                return None
            
            # Using ast.literal_eval to safely evaluate the Python dictionary
            results_dict_str = results_match.group(1)
            results_dict = ast.literal_eval(results_dict_str)
            
            # Look for the specific task
            if 'all' in results_dict and TASK_NAME in results_dict['all']:
                task_data = results_dict['all'][TASK_NAME]
                score = task_data.get('acc_norm,none')
            elif TASK_NAME in results_dict:
                task_data = results_dict[TASK_NAME]
                score = task_data.get('acc_norm,none')
            else:
                print(f"Task '{TASK_NAME}' not found for {model_name}")
                return None
            
            if score is not None:
                result = {
                    "model": model_name,
                    "score": score
                }
                
                # Cache the result
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                except Exception as e:
                    print(f"Error caching result for {model_name}: {e}")
                
                print(f"{model_name}: Score = {score}")
                return result
            
            return None
            
        except Exception as e:
            print(f"Error processing {model_name} (attempt {retry+1}/{MAX_RETRIES+1}): {e}")
            if retry == MAX_RETRIES:
                return None
            # Wait before retrying
            time.sleep(REQUEST_DELAY)

def load_resume_state():
    """Load resume state if it exists"""
    if os.path.exists(RESUME_FILE):
        try:
            with open(RESUME_FILE, 'r') as f:
                resume_state = json.load(f)
                print(f"Resuming from previous run. Already processed {len(resume_state['processed'])} models.")
                return resume_state
        except Exception as e:
            print(f"Error loading resume state: {e}")
    
    return {
        "processed": [],
        "qualifying_models": []
    }

def save_resume_state(processed, qualifying_models):
    """Save current progress for resume capability"""
    try:
        with open(RESUME_FILE, 'w') as f:
            json.dump({
                "processed": processed,
                "qualifying_models": qualifying_models
            }, f)
    except Exception as e:
        print(f"Error saving resume state: {e}")

def process_batch(models_batch, resume_state):
    """Process a batch of models"""
    qualifying_models = resume_state["qualifying_models"]
    processed = resume_state["processed"]
    
    # Filter out already processed models
    models_to_process = [m for m in models_batch if m not in processed]
    
    if not models_to_process:
        print("All models in this batch have already been processed.")
        return qualifying_models, processed
    
    # Process models with a progress bar
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_model_score, model): model for model in models_to_process}
        
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing models"):
                model = futures[future]
                result = future.result()
                processed.append(model)  # Mark as processed regardless of result
                
                if result and result["score"] > SCORE_THRESHOLD:
                    qualifying_models.append(result)
                
                # Save progress every 10 models
                if len(processed) % 10 == 0:
                    save_resume_state(processed, qualifying_models)
                    
        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")
            save_resume_state(processed, qualifying_models)
            print(f"Progress saved. Processed {len(processed)} models so far.")
            raise
    
    return qualifying_models, processed

def main(args):
    global REQUEST_DELAY, MAX_RETRIES, MAX_WORKERS
    
    # Update global settings from command line arguments
    REQUEST_DELAY = args.delay
    MAX_RETRIES = args.retries
    MAX_WORKERS = args.workers
    
    print(f"Filtering models with {TASK_NAME} score above {SCORE_THRESHOLD}...")
    print(f"Settings: delay={REQUEST_DELAY}s, retries={MAX_RETRIES}, workers={MAX_WORKERS}, batch_size={args.batch_size}")
    
    # Load resume state
    resume_state = load_resume_state()
    qualifying_models = resume_state["qualifying_models"]
    processed = resume_state["processed"]
    
    # Get full list of models to check
    all_models = get_model_list(prioritize_popular=args.prioritize)
    
    # Calculate remaining models
    remaining_models = [m for m in all_models if m not in processed]
    print(f"Total models: {len(all_models)}, Remaining: {len(remaining_models)}")
    
    # Process in batches to allow for easier interruption
    batch_size = args.batch_size
    try:
        for i in range(0, len(remaining_models), batch_size):
            batch = remaining_models[i:i+batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(remaining_models) + batch_size - 1)//batch_size}")
            qualifying_models, processed = process_batch(batch, {"qualifying_models": qualifying_models, "processed": processed})
            save_resume_state(processed, qualifying_models)
    except KeyboardInterrupt:
        print("\nScript interrupted. Progress has been saved.")
    
    # Sort by score in descending order
    qualifying_models.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\nFound {len(qualifying_models)} models with score > {SCORE_THRESHOLD}")
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(qualifying_models, f, indent=2)
    
    print(f"Results saved to {args.output}")
    
    # Display top 5 models
    if qualifying_models:
        print("\nTop 5 Models:")
        for i, model_data in enumerate(qualifying_models[:5], 1):
            print(f"{i}. {model_data['model']} - Score: {model_data['score']}")
    
    # If all models are processed, clean up resume file
    if len(processed) == len(all_models):
        if os.path.exists(RESUME_FILE):
            os.remove(RESUME_FILE)
            print("All models processed. Resume state file removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter models from the leaderboard based on their scores.")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, help="Delay between requests in seconds")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="Maximum number of retries for rate-limited requests")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Maximum number of concurrent threads")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing models")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file for qualifying models")
    parser.add_argument("--no-prioritize", dest="prioritize", action="store_false", help="Don't prioritize popular models")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    parser.set_defaults(prioritize=True)
    
    args = parser.parse_args()
    
    if args.reset and os.path.exists(RESUME_FILE):
        os.remove(RESUME_FILE)
        print("Reset progress. Starting fresh.")
    
    main(args)
