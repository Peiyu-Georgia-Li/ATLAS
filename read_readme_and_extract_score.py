from datasets import load_dataset
import requests

model_name= "0-hero/Matter-0.2-7B-DPO"



# Read the README.md file and extract specific score
def read_readme_and_extract_bbh_score():
    model_id = model_name.replace('/', '__')
    readme_url = f"https://huggingface.co/datasets/open-llm-leaderboard/{model_id}-details/resolve/main/README.md"
    try:
        response = requests.get(readme_url)
        if response.status_code == 200:
            readme_content = response.text
            
            # Extract score for the specific task
            import re
            import ast
            
            # Try to find the "Latest results" section with the URL and Python dictionary data
            # The pattern matches: ## Latest results followed by any text, URL, and then Python code block
            results_pattern = r'## Latest results[\s\S]*?\(.*?\):[\s\S]*?```python\s*({[\s\S]*?})\s*```'
            results_match = re.search(results_pattern, readme_content)
            
            if results_match:
                try:
                    # Using ast.literal_eval to safely evaluate the Python dictionary
                    results_dict_str = results_match.group(1)
                    results_dict = ast.literal_eval(results_dict_str)
                    
                    # Look for the specific task "leaderboard_bbh_tracking_shuffled_objects_five_objects"
                    # It might be nested within the 'all' dictionary
                    task_name = "leaderboard_bbh_tracking_shuffled_objects_five_objects"
                    
                    if 'all' in results_dict and task_name in results_dict['all']:
                        task_data = results_dict['all'][task_name]
                        print(f"\n=== SCORE FOR {task_name} ===")
                        print(f"Score (acc_norm,none): {task_data.get('acc_norm,none')}")
                        print(f"Standard Error: {task_data.get('acc_norm_stderr,none')}")
                        print(f"Alias: {task_data.get('alias')}")
                        print("=============================")
                    elif task_name in results_dict:
                        # Check if it's directly in the top level
                        task_data = results_dict[task_name]
                        print(f"\n=== SCORE FOR {task_name} ===")
                        print(f"Score (acc_norm,none): {task_data.get('acc_norm,none')}")
                        print(f"Standard Error: {task_data.get('acc_norm_stderr,none')}")
                        print(f"Alias: {task_data.get('alias')}")
                        print("=============================")
                    else:
                        print(f"\nTask '{task_name}' not found in the results.")
                        print("Available top-level keys:")
                        for key in results_dict.keys():
                            print(f"- {key}")
                        if 'all' in results_dict:
                            print("\nAvailable tasks in 'all':")
                            for key in results_dict['all'].keys():
                                print(f"- {key}")
                except (SyntaxError, ValueError) as e:
                    print(f"Failed to parse Python dictionary data from README: {e}")
                    print("First 200 characters of captured text:")
                    print(results_dict_str[:200] + "...")
            else:
                print("Could not find 'Latest results' section in README.")
                print("\n=== FULL README CONTENT ===")
                print(readme_content)
                print("=== END OF README ===")
        else:
            print(f"Failed to fetch README: HTTP {response.status_code}")
            print(f"README URL: {readme_url}")
    except Exception as e:
        print(f"Error reading README: {e}")



# Read the README.md file and extract specific score
def read_readme_and_extract_mmlu_score():
    model_id = model_name.replace('/', '__')
    readme_url = f"https://huggingface.co/datasets/open-llm-leaderboard/{model_id}-details/resolve/main/README.md"
    try:
        response = requests.get(readme_url)
        if response.status_code == 200:
            readme_content = response.text
            
            # Extract score for the specific task
            import re
            import ast
            
            # Try to find the "Latest results" section with the URL and Python dictionary data
            # The pattern matches: ## Latest results followed by any text, URL, and then Python code block
            results_pattern = r'## Latest results[\s\S]*?\(.*?\):[\s\S]*?```python\s*({[\s\S]*?})\s*```'
            results_match = re.search(results_pattern, readme_content)
            
            if results_match:
                try:
                    # Using ast.literal_eval to safely evaluate the Python dictionary
                    results_dict_str = results_match.group(1)
                    results_dict = ast.literal_eval(results_dict_str)
                    
                    # Look for the specific task "leaderboard_bbh_tracking_shuffled_objects_five_objects"
                    # It might be nested within the 'all' dictionary
                    task_name = "leaderboard_bbh_tracking_shuffled_objects_five_objects"
                    
                    if 'all' in results_dict and task_name in results_dict['all']:
                        task_data = results_dict['all'][task_name]
                        print(f"\n=== SCORE FOR {task_name} ===")
                        print(f"Score (acc_norm,none): {task_data.get('acc_norm,none')}")
                        print(f"Standard Error: {task_data.get('acc_norm_stderr,none')}")
                        print(f"Alias: {task_data.get('alias')}")
                        print("=============================")
                    elif task_name in results_dict:
                        # Check if it's directly in the top level
                        task_data = results_dict[task_name]
                        print(f"\n=== SCORE FOR {task_name} ===")
                        print(f"Score (acc_norm,none): {task_data.get('acc_norm,none')}")
                        print(f"Standard Error: {task_data.get('acc_norm_stderr,none')}")
                        print(f"Alias: {task_data.get('alias')}")
                        print("=============================")
                    else:
                        print(f"\nTask '{task_name}' not found in the results.")
                        print("Available top-level keys:")
                        for key in results_dict.keys():
                            print(f"- {key}")
                        if 'all' in results_dict:
                            print("\nAvailable tasks in 'all':")
                            for key in results_dict['all'].keys():
                                print(f"- {key}")
                except (SyntaxError, ValueError) as e:
                    print(f"Failed to parse Python dictionary data from README: {e}")
                    print("First 200 characters of captured text:")
                    print(results_dict_str[:200] + "...")
            else:
                print("Could not find 'Latest results' section in README.")
                print("\n=== FULL README CONTENT ===")
                print(readme_content)
                print("=== END OF README ===")
        else:
            print(f"Failed to fetch README: HTTP {response.status_code}")
            print(f"README URL: {readme_url}")
    except Exception as e:
        print(f"Error reading README: {e}")


# Call the function to read the README and extract score
read_readme_and_extract_bbh_score()
