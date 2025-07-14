# FINAL WORKING VERSION - Real Phi-2 responses from Open LLM Leaderboard
import datasets
import numpy as np

def show_real_model_responses(model_name, benchmark_name, split_name="latest"):
    """
    Show REAL model responses using the correct data structure
    
    Args:
        model_name: Model identifier (e.g., "microsoft/phi-2", "meta-llama/Llama-2-7b-hf")
        benchmark_name: Benchmark to examine (e.g., "harness_hellaswag_10", "harness_arc_challenge_25")
        split_name: Dataset split to use (default: "latest")
    """
    # Convert model name to dataset format
    dataset_path = f"open-llm-leaderboard-old/details_{model_name.replace('/', '__')}"
    
    print(f"🤖 {model_name.upper()} - REAL RESPONSES FROM OPEN LLM LEADERBOARD")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🎯 Benchmark: {benchmark_name}")
    print(f"📋 Split: {split_name}")
    print("=" * 80)
    
    try:
        # Load the dataset using the specified parameters
        dataset = datasets.load_dataset(dataset_path, benchmark_name, split=split_name)
        
        print(f"✅ Successfully loaded {len(dataset)} examples from {benchmark_name}")
        
        # Show first 3 real examples
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            
            print(f"\n📝 REAL QUESTION #{i+1}")
            print("=" * 60)
            
            # Extract the actual question from the 'example' field
            question_text = example.get('example', 'No question found')
            print(f"🔍 Context: {question_text}")
            
            # The predictions field contains the model's confidence scores for each choice
            predictions = example.get('predictions', [])
            
            if predictions:
                num_choices = len(predictions)
                print(f"\n🧠 {model_name}'s Confidence Scores:")
                choice_labels = ['A', 'B', 'C', 'D', 'E'][:num_choices]
                
                for j, (label, score) in enumerate(zip(choice_labels, predictions)):
                    marker = "👉" if j == np.argmax(predictions) else "  "
                    print(f"   {marker} {label}) Confidence: {score:.4f}")
                
                # Show which choice the model picked
                model_choice = np.argmax(predictions)
                print(f"\n🤖 {model_name}'s Choice: {choice_labels[model_choice]} (highest confidence)")
                
                # Check if it was correct (acc_norm field indicates correctness)
                accuracy = example.get('metrics', {}).get('acc_norm', example.get('acc_norm', 0))
                if accuracy == 1.0:
                    print(f"📊 Result: ✅ CORRECT")
                elif accuracy == 0.0:
                    print(f"📊 Result: ❌ INCORRECT") 
                else:
                    print(f"📊 Result: Unknown (score: {accuracy})")
            
            else:
                print(f"⚠️  No predictions found")
        
        # Calculate overall accuracy
        correct_count = 0
        total_count = len(dataset)
        
        for example in dataset:
            metrics = example.get('metrics', {})
            acc_norm = metrics.get('acc_norm', metrics.get('acc', 0))
            if acc_norm == 1.0:
                correct_count += 1
        
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"\n" + "=" * 80)
        print(f"📊 {model_name.upper()} OVERALL PERFORMANCE")
        print("=" * 80)
        print(f"Benchmark: {benchmark_name}")
        print(f"Total Questions: {total_count}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        return dataset  # Return dataset for further analysis
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"💡 Make sure the model '{model_name}' exists in the leaderboard")
        print(f"💡 Available benchmarks might include:")
        print(f"   • harness_hellaswag_10 (common sense reasoning)")
        print(f"   • harness_arc_challenge_25 (science questions)")  
        print(f"   • harness_mmlu_5 (general knowledge)")
        print(f"   • harness_truthfulqa_mc_0 (truthfulness)")
        print(f"   • harness_winogrande_5 (reading comprehension)")
        return None

def compare_with_original_hellaswag():
    """
    Show how the original HellaSwag questions look vs the evaluation format
    """
    print(f"\n🔍 COMPARING WITH ORIGINAL HELLASWAG FORMAT")
    print("=" * 80)
    
    try:
        # Load original HellaSwag dataset for comparison
        original = datasets.load_dataset("Rowan/hellaswag", split="validation")
        
        print("📋 Original HellaSwag Example:")
        example = original[0]
        activity = example.get('activity_label', '')
        context = example.get('ctx', '')
        endings = example.get('endings', [])
        correct = example.get('label', -1)
        
        print(f"🎯 Activity: {activity}")
        print(f"🔍 Context: {context}")
        print(f"📝 Endings:")
        for i, ending in enumerate(endings):
            marker = "✅" if i == correct else "🔘"
            print(f"   {marker} {i}) {ending}")
        
        print(f"\n💡 This shows what the original questions look like!")
        print(f"   The evaluation dataset contains processed versions of these.")
        
    except Exception as e:
        print(f"Could not load original HellaSwag: {e}")

def show_arc_examples(model_name, split_name="latest"):
    """
    Show ARC Challenge examples which have clearer question structure
    """
    print(f"\n🔬 {model_name.upper()} ON ARC CHALLENGE (SCIENCE QUESTIONS)")
    print("=" * 80)
    
    try:
        dataset_path = f"open-llm-leaderboard-old/details_{model_name.replace('/', '__')}"
        dataset = datasets.load_dataset(dataset_path, "harness_arc_challenge_25", split=split_name)
        
        print(f"✅ Successfully loaded {len(dataset)} ARC examples")
        
        # Show first 2 examples
        for i in range(min(2, len(dataset))):
            example = dataset[i]
            
            print(f"\n📝 REAL ARC QUESTION #{i+1}")
            print("-" * 60)
            
            # Try to extract the question from the full_prompt
            full_prompt = example.get('full_prompt', '')
            if full_prompt:
                # Look for the actual question in the prompt
                lines = full_prompt.split('\n')
                question_line = None
                for line in lines:
                    if '?' in line and len(line) > 20:  # Likely a question
                        question_line = line.strip()
                        break
                
                if question_line:
                    print(f"❓ Question: {question_line}")
                else:
                    # Fallback to the example field
                    print(f"❓ Context: {example.get('example', 'No question found')[:200]}...")
            
            # Show the predictions
            predictions = example.get('predictions', [])
            if predictions:
                print(f"\n🤖 {model_name}'s Confidence Scores:")
                choice_labels = ['A', 'B', 'C', 'D', 'E'][:len(predictions)]
                
                for j, (label, score) in enumerate(zip(choice_labels, predictions)):
                    marker = "👉" if j == np.argmax(predictions) else "  "
                    print(f"   {marker} {label}) {score:.4f}")
                
                model_choice = choice_labels[np.argmax(predictions)]
                print(f"\n🤖 {model_name}'s Answer: {model_choice}")
                
                # Check correctness
                accuracy = example.get('metrics', {}).get('acc_norm', 0)
                result = "✅ CORRECT" if accuracy == 1.0 else "❌ INCORRECT"
                print(f"📊 Result: {result}")
        
    except Exception as e:
        print(f"❌ Error loading ARC for {model_name}: {str(e)}")

def search_for_specific_content(model_name, benchmark_name, search_term="face", split_name="latest"):
    """
    Find examples containing specific content for any model/benchmark
    """
    print(f"\n🔍 SEARCHING {model_name.upper()} FOR '{search_term.upper()}'")
    print("=" * 80)
    
    try:
        dataset_path = f"open-llm-leaderboard-old/details_{model_name.replace('/', '__')}"
        dataset = datasets.load_dataset(dataset_path, benchmark_name, split=split_name)
        
        found_examples = []
        for i, example in enumerate(dataset):
            question_text = example.get('example', '').lower()
            if search_term.lower() in question_text:
                found_examples.append((i, example))
            
            if len(found_examples) >= 3:  # Show first 3 matches
                break
        
        if found_examples:
            print(f"✅ Found {len(found_examples)} examples:")
            
            for idx, (original_idx, example) in enumerate(found_examples):
                print(f"\n📍 Match {idx+1} (Question #{original_idx+1}):")
                question = example.get('example', 'No question')
                print(f"🔍 {question}")
                
                predictions = example.get('predictions', [])
                if predictions:
                    model_choice = chr(65 + np.argmax(predictions))  # Convert to A, B, C, D
                    confidence = max(predictions)
                    
                    accuracy = example.get('metrics', {}).get('acc_norm', 0)
                    result = "✅ CORRECT" if accuracy == 1.0 else "❌ INCORRECT"
                    
                    print(f"🤖 {model_name} chose: {model_choice} (confidence: {confidence:.3f}) - {result}")
        else:
            print(f"❌ No examples found containing '{search_term}'")
    
    except Exception as e:
        print(f"❌ Search error: {str(e)}")

if __name__ == "__main__":
    # ======================================================================
    # 🎯 CONFIGURATION - CHANGE THESE TO TEST DIFFERENT MODELS/BENCHMARKS
    # ======================================================================
    
    # Model to analyze (replace with any model from the leaderboard)
    MODEL_NAME = "microsoft/phi-2"  # Change this to test other models!
    # Other examples:
    # MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    # MODEL_NAME = "mistralai/Mistral-7B-v0.1" 
    # MODEL_NAME = "teknium/OpenHermes-2.5-Mistral-7B"
    
    # Primary benchmark to analyze
    PRIMARY_BENCHMARK = "harness_hellaswag_10"  # Change this to test other benchmarks!
    # Other available benchmarks:
    # PRIMARY_BENCHMARK = "harness_arc_challenge_25"    # Science reasoning
    # PRIMARY_BENCHMARK = "harness_mmlu_5"              # General knowledge  
    # PRIMARY_BENCHMARK = "harness_truthfulqa_mc_0"     # Truthfulness
    # PRIMARY_BENCHMARK = "harness_winogrande_5"        # Reading comprehension
    # PRIMARY_BENCHMARK = "harness_gsm8k_5"             # Math word problems
    
    # Dataset split (usually "latest")
    SPLIT_NAME = "latest"
    
    # Search terms to look for
    SEARCH_TERMS = ["washing", "face", "water"]  # Change these to search for different topics
    
    # ======================================================================
    # 🚀 ANALYSIS EXECUTION
    # ======================================================================
    
    print(f"🔬 ANALYZING MODEL: {MODEL_NAME}")
    print(f"📊 PRIMARY BENCHMARK: {PRIMARY_BENCHMARK}")
    print("=" * 100)
    
    # Main analysis
    dataset = show_real_model_responses(MODEL_NAME, PRIMARY_BENCHMARK, SPLIT_NAME)
    
    # Show original format comparison (only for HellaSwag)
    if "hellaswag" in PRIMARY_BENCHMARK.lower():
        compare_with_original_hellaswag()
    
    # Show ARC examples if not already the primary benchmark
    if "arc" not in PRIMARY_BENCHMARK.lower():
        show_arc_examples(MODEL_NAME, SPLIT_NAME)
    
    # Search for specific content
    for search_term in SEARCH_TERMS:
        search_for_specific_content(MODEL_NAME, PRIMARY_BENCHMARK, search_term, SPLIT_NAME)
    
    print(f"\n" + "=" * 100)
    print("🎯 SUMMARY")
    print("=" * 100)
    print(f"✅ Analyzed {MODEL_NAME} on {PRIMARY_BENCHMARK}")
    print(f"📊 This shows REAL responses from the Open LLM Leaderboard!")
    print(f"🔧 To test other models/benchmarks, change the variables in __main__")
    print(f"💡 Available models: Any model that has been evaluated on the leaderboard")
    print(f"📋 Available benchmarks: harness_hellaswag_10, harness_arc_challenge_25, etc.")