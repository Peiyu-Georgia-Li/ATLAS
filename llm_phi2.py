# FINAL WORKING VERSION - Real Phi-2 responses from Open LLM Leaderboard
import datasets
import numpy as np

def show_real_phi2_responses():
    """
    Show REAL Microsoft Phi-2 responses using the correct data structure
    """
    print("🤖 MICROSOFT PHI-2 - REAL RESPONSES FROM OPEN LLM LEADERBOARD")
    print("=" * 80)
    
    try:
        # Load the dataset using the correct path and split
        dataset = datasets.load_dataset(
            "open-llm-leaderboard-old/details_microsoft__phi-2", 
            "harness_hellaswag_10", 
            split="latest"
        )
        
        print(f"✅ Successfully loaded {len(dataset)} HellaSwag examples")
        
        # Show first 3 real examples
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            
            print(f"\n📝 REAL HELLASWAG QUESTION #{i+1}")
            print("=" * 60)
            
            # Extract the actual question from the 'example' field
            question_text = example.get('example', 'No question found')
            print(f"🔍 Context: {question_text}")
            
            # The predictions field contains the model's confidence scores for each choice
            predictions = example.get('predictions', [])
            
            if predictions and len(predictions) == 4:  # HellaSwag has 4 choices
                print(f"\n🧠 Phi-2's Confidence Scores:")
                choice_labels = ['A', 'B', 'C', 'D']
                
                for j, (label, score) in enumerate(zip(choice_labels, predictions)):
                    marker = "👉" if j == np.argmax(predictions) else "  "
                    print(f"   {marker} {label}) Confidence: {score:.4f}")
                
                # Show which choice Phi-2 picked
                phi2_choice = np.argmax(predictions)
                print(f"\n🤖 Phi-2's Choice: {choice_labels[phi2_choice]} (highest confidence)")
                
                # Check if it was correct (acc_norm field indicates correctness)
                accuracy = example.get('metrics', {}).get('acc_norm', example.get('acc_norm', 0))
                if accuracy == 1.0:
                    print(f"📊 Result: ✅ CORRECT")
                elif accuracy == 0.0:
                    print(f"📊 Result: ❌ INCORRECT") 
                else:
                    print(f"📊 Result: Unknown (score: {accuracy})")
            
            else:
                print(f"⚠️  Unexpected prediction format: {predictions}")
        
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
        print(f"📊 PHI-2 OVERALL PERFORMANCE ON HELLASWAG")
        print("=" * 80)
        print(f"Total Questions: {total_count}")
        print(f"Correct Answers: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

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

def show_arc_examples():
    """
    Show ARC Challenge examples which have clearer question structure
    """
    print(f"\n🔬 PHI-2 ON ARC CHALLENGE (SCIENCE QUESTIONS)")
    print("=" * 80)
    
    try:
        dataset = datasets.load_dataset(
            "open-llm-leaderboard-old/details_microsoft__phi-2", 
            "harness_arc_challenge_25", 
            split="latest"
        )
        
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
                print(f"\n🤖 Phi-2's Confidence Scores:")
                choice_labels = ['A', 'B', 'C', 'D', 'E'][:len(predictions)]
                
                for j, (label, score) in enumerate(zip(choice_labels, predictions)):
                    marker = "👉" if j == np.argmax(predictions) else "  "
                    print(f"   {marker} {label}) {score:.4f}")
                
                phi2_choice = choice_labels[np.argmax(predictions)]
                print(f"\n🤖 Phi-2's Answer: {phi2_choice}")
                
                # Check correctness
                accuracy = example.get('metrics', {}).get('acc_norm', 0)
                result = "✅ CORRECT" if accuracy == 1.0 else "❌ INCORRECT"
                print(f"📊 Result: {result}")
        
    except Exception as e:
        print(f"❌ Error loading ARC: {str(e)}")

def search_for_specific_content(search_term="face"):
    """
    Find examples containing specific content
    """
    print(f"\n🔍 SEARCHING FOR QUESTIONS CONTAINING '{search_term.upper()}'")
    print("=" * 80)
    
    try:
        dataset = datasets.load_dataset(
            "open-llm-leaderboard-old/details_microsoft__phi-2", 
            "harness_hellaswag_10", 
            split="latest"
        )
        
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
                    phi2_choice = chr(65 + np.argmax(predictions))  # Convert to A, B, C, D
                    confidence = max(predictions)
                    
                    accuracy = example.get('metrics', {}).get('acc_norm', 0)
                    result = "✅ CORRECT" if accuracy == 1.0 else "❌ INCORRECT"
                    
                    print(f"🤖 Phi-2 chose: {phi2_choice} (confidence: {confidence:.3f}) - {result}")
        else:
            print(f"❌ No examples found containing '{search_term}'")
    
    except Exception as e:
        print(f"❌ Search error: {str(e)}")

if __name__ == "__main__":
    # Show real Phi-2 responses
    show_real_phi2_responses()
    
    # Compare with original format
    compare_with_original_hellaswag()
    
    # Show ARC examples
    show_arc_examples()
    
    # Search for specific content
    search_for_specific_content("washing")
    search_for_specific_content("face")
    
    print(f"\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print("✅ This shows REAL Microsoft Phi-2 responses from the Open LLM Leaderboard!")
    print("📊 You can see exactly how Phi-2 answered each question")
    print("🔍 The 'example' field contains the actual questions asked")
    print("🤖 The 'predictions' field shows Phi-2's confidence for each choice")
    print("📈 The 'metrics' field indicates whether Phi-2 got it right or wrong")