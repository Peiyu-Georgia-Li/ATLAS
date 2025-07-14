# Debug script to understand the exact data structure
import datasets
import json

def deep_examine_structure():
    """Thoroughly examine what's actually in the dataset"""
    print("🔍 DEEP EXAMINATION OF PHI-2 DATA STRUCTURE")
    print("=" * 80)
    
    try:
        # Load the dataset
        dataset = datasets.load_dataset(
            "open-llm-leaderboard-old/details_microsoft__phi-2", 
            "harness_hellaswag_10"
        )
        
        print(f"Available splits: {list(dataset.keys())}")
        
        # Use latest split
        split_name = "latest" if "latest" in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        
        print(f"\nUsing split: {split_name}")
        print(f"Number of examples: {len(data)}")
        
        # Examine first example in extreme detail
        example = data[0]
        print(f"\n📊 COMPLETE FIRST EXAMPLE:")
        print("=" * 60)
        
        def print_nested(obj, indent=0):
            """Recursively print nested data structures"""
            spaces = "  " * indent
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (dict, list)):
                        print(f"{spaces}{key}: {type(value).__name__}")
                        print_nested(value, indent + 1)
                    else:
                        # Truncate long strings
                        if isinstance(value, str) and len(value) > 100:
                            preview = value[:100] + "..."
                        else:
                            preview = value
                        print(f"{spaces}{key}: {preview}")
            elif isinstance(obj, list):
                print(f"{spaces}List with {len(obj)} items:")
                for i, item in enumerate(obj[:3]):  # Show first 3 items
                    print(f"{spaces}[{i}]:")
                    print_nested(item, indent + 1)
                if len(obj) > 3:
                    print(f"{spaces}... and {len(obj) - 3} more items")
            else:
                print(f"{spaces}{obj}")
        
        print_nested(example)
        
        # Try to find any field that might contain the actual question data
        print(f"\n🔍 SEARCHING FOR QUESTION DATA:")
        print("=" * 60)
        
        def search_for_text(obj, path=""):
            """Search for fields that might contain question text"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if isinstance(value, str) and len(value) > 20:
                        print(f"Found text in {new_path}: {value[:100]}...")
                    elif isinstance(value, (dict, list)):
                        search_for_text(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj[:2]):  # Check first 2 items
                    new_path = f"{path}[{i}]"
                    search_for_text(item, new_path)
        
        search_for_text(example)
        
        # Look at a few more examples to see if the structure is consistent
        print(f"\n🔍 CHECKING CONSISTENCY ACROSS EXAMPLES:")
        print("=" * 60)
        
        for i in range(min(3, len(data))):
            ex = data[i]
            print(f"\nExample {i} keys: {list(ex.keys()) if isinstance(ex, dict) else type(ex)}")
            
            # Look for any fields that might have the question content
            for key in ex.keys() if isinstance(ex, dict) else []:
                value = ex[key]
                if isinstance(value, str) and 10 < len(value) < 200:
                    print(f"  {key}: {value}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def try_different_approach():
    """Try a completely different approach to access the data"""
    print(f"\n🔄 TRYING ALTERNATIVE DATA ACCESS")
    print("=" * 80)
    
    try:
        # Maybe try the new leaderboard format
        new_dataset = datasets.load_dataset(
            "open-llm-leaderboard/details_microsoft__phi-2", 
            "harness_hellaswag_10"
        )
        
        print("✅ Found data in new format!")
        split_name = "latest" if "latest" in new_dataset else list(new_dataset.keys())[0]
        data = new_dataset[split_name]
        
        print(f"New format - Split: {split_name}, Examples: {len(data)}")
        
        example = data[0]
        print(f"New format example keys: {list(example.keys())}")
        
        # Print first example from new format
        print(f"\n📊 NEW FORMAT EXAMPLE:")
        for key, value in example.items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {preview}")
            else:
                print(f"  {key}: {type(value)} - {value}")
        
    except Exception as e:
        print(f"New format not available: {e}")
    
    # Try to load the raw HellaSwag dataset to compare structure
    try:
        print(f"\n🔄 CHECKING ORIGINAL HELLASWAG DATASET")
        print("-" * 60)
        
        original_hellaswag = datasets.load_dataset("Rowan/hellaswag", split="validation")
        
        print(f"Original HellaSwag example:")
        orig_example = original_hellaswag[0]
        for key, value in orig_example.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Could not load original HellaSwag: {e}")

def manual_data_inspection():
    """Manually inspect the data to find the right fields"""
    print(f"\n🕵️ MANUAL DATA INSPECTION")
    print("=" * 80)
    
    try:
        dataset = datasets.load_dataset(
            "open-llm-leaderboard-old/details_microsoft__phi-2", 
            "harness_hellaswag_10"
        )
        
        split_name = "latest" if "latest" in dataset else list(dataset.keys())[0]
        data = dataset[split_name]
        
        # Get raw example and convert to dict for easier inspection
        example = data[0]
        
        # Try to convert to JSON for better readability
        print("📄 RAW EXAMPLE AS JSON:")
        try:
            json_str = json.dumps(dict(example), indent=2, default=str)
            # Print first 2000 characters
            print(json_str[:2000])
            if len(json_str) > 2000:
                print("... (truncated)")
        except Exception as e:
            print(f"Could not convert to JSON: {e}")
            print(f"Raw example: {example}")
        
        # Look for specific patterns that might indicate question data
        print(f"\n🔍 LOOKING FOR QUESTION PATTERNS:")
        
        # Common patterns in evaluation datasets
        patterns_to_check = [
            'question', 'prompt', 'input', 'context', 'ctx', 'stem',
            'choices', 'options', 'endings', 'completions',
            'answer', 'label', 'target', 'gold'
        ]
        
        def find_patterns(obj, path="", max_depth=3):
            if max_depth <= 0:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if key matches our patterns
                    if any(pattern in key.lower() for pattern in patterns_to_check):
                        print(f"🎯 Found pattern in {current_path}: {type(value)} = {value}")
                    
                    # Recurse into nested structures
                    if isinstance(value, (dict, list)):
                        find_patterns(value, current_path, max_depth - 1)
            
            elif isinstance(obj, list) and len(obj) > 0:
                # Check first item in list
                find_patterns(obj[0], f"{path}[0]", max_depth - 1)
        
        find_patterns(example)
        
    except Exception as e:
        print(f"❌ Manual inspection failed: {e}")

if __name__ == "__main__":
    print("🕵️ PHI-2 DATA STRUCTURE DETECTIVE")
    print("Let's figure out exactly how this data is structured!")
    print("=" * 80)
    
    # Step 1: Deep examination
    data = deep_examine_structure()
    
    # Step 2: Try different approaches
    try_different_approach()
    
    # Step 3: Manual inspection
    manual_data_inspection()
    
    print(f"\n" + "=" * 80)
    print("🎯 NEXT STEPS:")
    print("Based on the output above, we can build the correct extraction logic.")
    print("Look for fields that contain question text, choices, and answers.")