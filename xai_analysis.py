# xai_analysis.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import os
import lime
import lime.lime_text
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================
DATASET_CONFIG = {
    "name": "go_emotions",
    "subset": "simplified",  # or "raw" for 28 classes
    "split_names": {"train": "train", "validation": "validation", "test": "test"},
    "text_column": "text",
    "label_column": "labels",
}

# Use the trained weighted model (best for XAI analysis)
MODEL_PATH = "./outputs/weighted_model_go-emotions/final_model"
LIME_OUTPUT_DIR = "./outputs/weighted_model_go-emotions/lime_explanations"

os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"XAI Analysis with LIME")
print(f"{'='*60}")
print(f"Dataset: {DATASET_CONFIG['name']}")
print(f"Model path: {MODEL_PATH}")
print(f"Output dir: {LIME_OUTPUT_DIR}\n")


# ============================================================
# 1. EXTRACT LABEL NAMES FROM DATASET
# ============================================================
print("--- Loading Dataset to Extract Label Names ---")

if "subset" in DATASET_CONFIG:
    dataset = load_dataset(DATASET_CONFIG["name"], DATASET_CONFIG["subset"])
else:
    dataset = load_dataset(DATASET_CONFIG["name"])

label_column = DATASET_CONFIG["label_column"]
train_split = dataset[DATASET_CONFIG["split_names"]["train"]]

# Detect label structure
if hasattr(train_split.features[label_column], "names"):
    label_names = train_split.features[label_column].names
    is_multilabel = False
    print(f"✓ Single-label: {len(label_names)} classes")
    print(f"  Classes: {label_names}")
    
elif hasattr(train_split.features[label_column].feature, "names"):
    label_names = train_split.features[label_column].feature.names
    is_multilabel = True
    print(f"✓ Multi-label: {len(label_names)} classes")
    print(f"  Classes: {label_names[:10]}... (showing first 10)")
else:
    raise ValueError("Cannot determine label structure!")


# ============================================================
# 2. DEVICE AND MODEL LOADING
# ============================================================
print("\n--- Loading Model ---")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists(MODEL_PATH):
    print(f"✗ ERROR: Model not found at '{MODEL_PATH}'")
    print(f"Make sure you ran weighted_emotions.py first!")
    print(f"Expected path: {MODEL_PATH}")
    exit()

# PEFT support for LoRA adapters
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠ PEFT not available, will try loading as regular model")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Check if this is a LoRA/PEFT adapter
    adapter_config_path = os.path.join(MODEL_PATH, "adapter_config.json")
    if os.path.exists(adapter_config_path) and PEFT_AVAILABLE:
        print(f"  Loading as PEFT adapter...")
        peft_config = PeftConfig.from_pretrained(MODEL_PATH)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=len(label_names),
            problem_type="multi_label_classification" if is_multilabel else "single_label_classification",
            attn_implementation="eager"
        )
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        model = model.merge_and_unload()  # Merge LoRA weights for inference
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            attn_implementation="eager"
        )
    
    model.eval()
    model.to(device)
    print(f"✓ Model loaded from: {MODEL_PATH}")
    
    # Verify label mappings
    model.config.id2label = {id: label for id, label in enumerate(label_names)}
    model.config.label2id = {label: id for id, label in enumerate(label_names)}
    print(f"✓ Label mappings configured: {len(label_names)} labels")
    
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    exit()


# ============================================================
# 3. PREDICTOR FUNCTION FOR LIME
# ============================================================
def predictor(texts):
    """
    Predictor function for LIME.
    Returns probability distribution across classes.
    """
    inputs = tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    if is_multilabel:
        # Multi-label: use sigmoid
        probabilities = torch.sigmoid(logits)
    else:
        # Single-label: use softmax
        probabilities = torch.softmax(logits, dim=-1)
    
    return probabilities.cpu().numpy()


# ============================================================
# 4. INITIALIZE LIME EXPLAINER
# ============================================================
print("\n--- Initializing LIME Text Explainer ---")

explainer = lime.lime_text.LimeTextExplainer(
    class_names=label_names,
    random_state=42
)
print(f"✓ LIME explainer initialized with {len(label_names)} classes")


# ============================================================
# 5. SAMPLE TEXTS FOR ANALYSIS
# ============================================================
# Use different samples based on dataset
if DATASET_CONFIG['name'] == 'go_emotions':
    sample_texts_lime = [
        'I am so happy today, the sun is shining brightly!',
        'This news truly made me feel incredibly sad and heartbroken.',
        'Feeling absolutely delighted with the progress and achievements.',
        'I am furious about this unexpected error and the delay it caused.',
        'I was really surprised by the sudden turn of events.',
        'I have a profound love for nature and all its beauty.'
    ]
else:  # emotion dataset
    sample_texts_lime = [
        'I am so happy today, the sun is shining brightly!',
        'This news truly made me feel incredibly sad and heartbroken.',
        'Feeling absolutely delighted with the progress and achievements.',
        'I am furious about this unexpected error and the delay it caused.',
        'I was really surprised by the sudden turn of events.',
        'I have a profound love for nature and all its beauty.'
    ]


# ============================================================
# 6. LIME ANALYSIS
# ============================================================
print(f"\n--- Starting XAI Analysis with LIME (Multi-label: {is_multilabel}) ---\n")

for idx, text in enumerate(sample_texts_lime):
    print(f"\n{'='*80}")
    print(f"Example {idx+1}: '{text}'")
    print(f"{'='*80}")
    
    # Get model prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    if is_multilabel:
        probabilities = torch.sigmoid(logits)[0]
        # For multi-label, show predictions above threshold
        threshold = 0.5
        predicted_ids = (probabilities > threshold).nonzero(as_tuple=True)[0]
        predicted_labels = [model.config.id2label[id.item()] for id in predicted_ids]
        
        if len(predicted_labels) == 0:
            # If no predictions above threshold, take top-3
            top_ids = torch.topk(probabilities, k=min(3, len(label_names)))[1]
            predicted_labels = [model.config.id2label[id.item()] for id in top_ids]
        
        print(f"\nPredicted labels (top): {', '.join(predicted_labels)}")
        print("\nProbabilities for all classes (top 10):")
        sorted_probs = sorted(
            [(model.config.id2label[i], prob.item()) for i, prob in enumerate(probabilities)],
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in sorted_probs[:10]:
            print(f"  {label}: {prob:.4f}")
    
    else:
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = torch.argmax(probabilities).item()
        predicted_label = model.config.id2label[predicted_id]
        
        print(f"\nPredicted label: {predicted_label}")
        print("\nProbabilities for all classes:")
        sorted_probs = sorted(
            [(model.config.id2label[i], prob.item()) for i, prob in enumerate(probabilities)],
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in sorted_probs:
            print(f"  {label}: {prob:.4f}")
    
    # Generate LIME explanation
    print("\nGenerating LIME explanation...")
    explanation = explainer.explain_instance(
        text,
        predictor,
        top_labels=1 if not is_multilabel else 3,
        num_features=10,
        num_samples=2000
    )
    
    # Extract and display explanations
    if explanation.local_exp:
        for label_id in list(explanation.local_exp.keys())[:3]:  # Show top 3
            label_name = model.config.id2label[label_id]
            print(f"\n  Features for '{label_name}':")
            
            for word, weight in explanation.as_list(label=label_id):
                direction = "↑" if weight > 0 else "↓"
                print(f"    {direction} '{word}': {weight:.4f}")
        
        # Save HTML explanation
        html_path = os.path.join(LIME_OUTPUT_DIR, f"lime_explanation_{idx+1:02d}.html")
        with open(html_path, "w") as f:
            f.write(explanation.as_html())
        print(f"\n  ✓ Interactive explanation saved to: {html_path}")
    
    else:
        print("  ⚠ LIME could not generate explanation for this sample")


# ============================================================
# 7. SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("✓ XAI Analysis Complete")
print(f"{'='*80}")
print(f"\nInteractive LIME explanations saved to: {LIME_OUTPUT_DIR}/")
print("Open the .html files in your browser to explore the explanations interactively.")
print(f"\nModel used: {MODEL_PATH}")
print(f"Dataset: {DATASET_CONFIG['name']}")
print(f"Number of labels: {len(label_names)}")
print(f"Multi-label: {is_multilabel}")

# Save XAI summary to JSON
import json
xai_summary = {
    "model_path": MODEL_PATH,
    "dataset": DATASET_CONFIG["name"],
    "num_labels": len(label_names),
    "is_multilabel": is_multilabel,
    "num_examples_analyzed": len(sample_texts_lime),
    "explanations_dir": LIME_OUTPUT_DIR,
    "label_names": list(label_names)
}
xai_summary_path = os.path.join(LIME_OUTPUT_DIR, "xai_summary.json")
with open(xai_summary_path, 'w') as f:
    json.dump(xai_summary, f, indent=4)
print(f"✓ XAI summary saved to: {xai_summary_path}")

