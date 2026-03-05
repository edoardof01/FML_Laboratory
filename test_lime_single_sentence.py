#!/usr/bin/env python3
"""
Quick test script to run LIME on a single sentence
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import lime
import lime.lime_text
import os

# Configuration
MODEL_PATH = "./outputs/weighted_model_go-emotions/final_model"
OUTPUT_DIR = "./lime_test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test sentence
TEST_SENTENCE = "I am so happy and excited about this amazing day!"

print("=" * 80)
print("LIME Analysis - Single Sentence Test")
print("=" * 80)
print(f"\nTest sentence: '{TEST_SENTENCE}'")
print(f"Model: {MODEL_PATH}")
print(f"Output: {OUTPUT_DIR}")

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nLoading model on {device}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTrying alternative models...")
    # Try other models as fallback
    alternative_paths = [
        "./outputs/clean_baseline_go-emotions/final_model",
        "./outputs/go-emotions/final_model",
        "./kfold_model_go-emotions/final_distilbert_model_kfold"
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(alt_path):
            print(f"  Trying {alt_path}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(alt_path)
                model = AutoModelForSequenceClassification.from_pretrained(alt_path)
                model.to(device)
                model.eval()
                MODEL_PATH = alt_path
                print(f"  ✓ Successfully loaded from {alt_path}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
    else:
        print("\n✗ Could not load any model. Please check your model paths.")
        exit(1)

# Get label names
label_names = [model.config.id2label[i] for i in sorted(model.config.id2label.keys())]
print(f"\nModel has {len(label_names)} classes: {label_names[:5]}...")

# Define predictor function for LIME
def predictor(texts):
    """Predictor function for LIME"""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # Determine if multi-label based on problem_type
    is_multilabel = (hasattr(model.config, 'problem_type') and 
                     model.config.problem_type == "multi_label_classification")
    
    if is_multilabel:
        probabilities = torch.sigmoid(logits).cpu().numpy()
    else:
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    
    return probabilities

# Make prediction
print("\n" + "=" * 80)
print("Model Prediction")
print("=" * 80)

inputs = tokenizer(TEST_SENTENCE, return_tensors="pt", truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits

# Check if multi-label
is_multilabel = (hasattr(model.config, 'problem_type') and 
                 model.config.problem_type == "multi_label_classification")

if is_multilabel:
    probabilities = torch.sigmoid(logits)[0]
    threshold = 0.5
    predicted_ids = (probabilities > threshold).nonzero(as_tuple=True)[0]
    
    if len(predicted_ids) == 0:
        # Take top-3 if none above threshold
        top_ids = torch.topk(probabilities, k=3)[1]
        predicted_labels = [label_names[id.item()] for id in top_ids]
    else:
        predicted_labels = [label_names[id.item()] for id in predicted_ids]
    
    print(f"\nPredicted emotions: {', '.join(predicted_labels)}")
    print("\nTop 5 probabilities:")
    sorted_probs = sorted(
        [(label_names[i], prob.item()) for i, prob in enumerate(probabilities)],
        key=lambda x: x[1],
        reverse=True
    )
    for label, prob in sorted_probs[:5]:
        print(f"  {label}: {prob:.4f}")
else:
    probabilities = torch.softmax(logits, dim=-1)[0]
    predicted_id = torch.argmax(probabilities).item()
    predicted_label = label_names[predicted_id]
    
    print(f"\nPredicted emotion: {predicted_label}")
    print("\nTop 5 probabilities:")
    sorted_probs = sorted(
        [(label_names[i], prob.item()) for i, prob in enumerate(probabilities)],
        key=lambda x: x[1],
        reverse=True
    )
    for label, prob in sorted_probs[:5]:
        print(f"  {label}: {prob:.4f}")

# Initialize LIME explainer
print("\n" + "=" * 80)
print("LIME Explanation")
print("=" * 80)
print("\nGenerating LIME explanation (this may take a minute)...")

explainer = lime.lime_text.LimeTextExplainer(class_names=label_names)

# Generate explanation
explanation = explainer.explain_instance(
    TEST_SENTENCE,
    predictor,
    top_labels=3,  # Explain top 3 classes
    num_features=10,  # Show top 10 most important words
    num_samples=1000  # Number of samples for LIME
)

# Display explanation
print("\nWord importance for top predicted classes:")
print("-" * 80)

for label_id in list(explanation.local_exp.keys())[:3]:
    label_name = label_names[label_id]
    print(f"\n{label_name}:")
    
    for word, weight in explanation.as_list(label=label_id):
        direction = "↑ POSITIVE" if weight > 0 else "↓ NEGATIVE"
        print(f"  {direction:12} '{word}': {weight:+.4f}")

# Save HTML visualization
html_path = os.path.join(OUTPUT_DIR, "lime_explanation.html")
with open(html_path, "w") as f:
    f.write(explanation.as_html())

print("\n" + "=" * 80)
print("Output Files")
print("=" * 80)
print(f"\n✓ Interactive HTML visualization saved to:")
print(f"  {html_path}")
print(f"\nTo view the visualization, open the HTML file in your browser:")
print(f"  firefox {html_path}")
print(f"  or")
print(f"  google-chrome {html_path}")

# Also save as PNG if matplotlib is available
try:
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    # Get explanation for top label
    top_label_id = list(explanation.local_exp.keys())[0]
    top_label_name = label_names[top_label_id]
    
    # Get word importances
    word_importances = explanation.as_list(label=top_label_id)
    words = [w for w, _ in word_importances]
    weights = [w for _, w in word_importances]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = ['green' if w > 0 else 'red' for w in weights]
    ax.barh(words, weights, color=colors_list, alpha=0.7)
    ax.set_xlabel('Importance Weight', fontsize=12)
    ax.set_title(f'LIME Explanation - {top_label_name}\n"{TEST_SENTENCE}"', 
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    
    png_path = os.path.join(OUTPUT_DIR, "lime_explanation.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ PNG visualization saved to:")
    print(f"  {png_path}")
    
except ImportError:
    print("\n(matplotlib not available - skipping PNG generation)")

print("\n" + "=" * 80)
print("✓ LIME Analysis Complete!")
print("=" * 80)
