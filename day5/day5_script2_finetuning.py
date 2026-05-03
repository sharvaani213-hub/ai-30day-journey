# -*- coding: utf-8 -*-
# ============================================================
# DAY 5 -- SCRIPT 2: Fine-tuning LLMs
# Topics: what is fine-tuning, when to use it, LoRA, PEFT,
#         dataset preparation, training loop concepts,
#         full fine-tune vs LoRA comparison
# ============================================================

import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("DAY 5 -- SCRIPT 2: Fine-tuning LLMs")
print("=" * 60)

print("""
WHAT IS FINE-TUNING?
--------------------
Pre-trained LLMs (GPT, BERT, LLaMA) are trained on general internet text.
Fine-tuning = further train them on YOUR specific data.

Like this:
  GPT-4 knows everything generally
  Fine-tuned GPT-4 knows YOUR company's products, tone, policies

When to use fine-tuning vs RAG:

  USE RAG when:                    USE Fine-tuning when:
  ----------------------           ----------------------
  Data changes frequently          Data is relatively stable
  Need source citations            Need specific tone/style
  Quick to set up                  Need faster inference
  Low cost                         Have lots of labeled data
  External documents               Want model behavior change

FINE-TUNING METHODS:
--------------------
  1. Full fine-tune  -> update ALL parameters (expensive, needs A100 GPU)
  2. LoRA            -> update only small adapter matrices (runs on free Colab!)
  3. QLoRA           -> LoRA + 4-bit quantization (even more efficient)
  4. PEFT            -> umbrella term for efficient fine-tuning methods
""")


# -- SECTION 1: Understanding Model Parameters ----------------
print("SECTION 1: Model Parameters -- What Gets Updated?")
print("-" * 40)

model_sizes = {
    "BERT-base"    : 110,
    "GPT-2"        : 117,
    "LLaMA-7B"     : 7000,
    "LLaMA-13B"    : 13000,
    "GPT-3.5"      : 175000,
    "GPT-4"        : 1000000,  # estimated
}

print(f"\n  {'Model':<15} {'Params (M)':>12} {'Full FT VRAM':>13} {'LoRA VRAM':>12}")
print(f"  {'─'*15} {'─'*12} {'─'*13} {'─'*12}")

for model, params_m in model_sizes.items():
    # Full fine-tune needs ~16 bytes per param (model + gradients + optimizer)
    full_vram_gb = round(params_m * 16 / 1000, 1)
    # LoRA only needs ~2-4 bytes per param for adapters (1% of params)
    lora_vram_gb = round(params_m * 0.01 * 4 / 1000 + params_m * 2 / 1000, 1)

    full_note = "A100 needed" if full_vram_gb > 24 else f"{full_vram_gb}GB"
    lora_note = "Free Colab!" if lora_vram_gb < 16 else f"{lora_vram_gb}GB"

    print(f"  {model:<15} {params_m:>10,}M {full_note:>13} {lora_note:>12}")

print("""
  Key insight: LoRA lets you fine-tune a 7B model on a FREE Google Colab GPU!
  Full fine-tuning the same model would need an expensive A100 GPU (~$3/hour).
""")


# -- SECTION 2: LoRA -- How It Works --------------------------
print("\nSECTION 2: LoRA -- Low Rank Adaptation")
print("-" * 40)
print("""
LoRA = Low-Rank Adaptation of Large Language Models (2021, Microsoft)

The BIG idea:
  Instead of updating the full weight matrix W (huge!)
  We learn two tiny matrices A and B such that:

  W_updated = W_original + (B x A)

  Where:
    W is (4096 x 4096) = 16 million parameters  <- frozen, not updated
    A is (4096 x 8)    = 32,768 parameters       <- trained
    B is (8 x 4096)    = 32,768 parameters       <- trained

  Total trainable params: 65,536  vs  16,000,000
  That's 244x fewer parameters to train!

The 8 above is called the "rank" (r).
Lower rank = faster training, less memory, slightly lower quality.
Higher rank = better quality, more memory.
Common values: r=4, r=8, r=16, r=64
""")

# Demonstrate the math
print("  LoRA math demonstration:")
np.random.seed(42)

W = np.random.randn(10, 10)   # original weight matrix (frozen)
r = 2                          # rank
A = np.random.randn(10, r) * 0.01   # trained (initialized small)
B = np.zeros((r, 10))               # trained (initialized to zero)

# At start: B is zero so W_updated = W_original (no change)
W_updated = W + B.T @ A.T

print(f"  Original W shape    : {W.shape} = {W.size} params")
print(f"  LoRA A shape        : {A.shape} = {A.size} params")
print(f"  LoRA B shape        : {B.shape} = {B.size} params")
print(f"  LoRA total params   : {A.size + B.size} (vs {W.size} full)")
print(f"  Compression ratio   : {W.size / (A.size + B.size):.1f}x fewer params")
print(f"  W_updated = W + B.T @ A.T, shape: {W_updated.shape}")
print(f"  At init: W_updated == W_original: {np.allclose(W, W_updated)}")


# -- SECTION 3: Dataset Preparation ---------------------------
print("\n\nSECTION 3: Dataset Preparation for Fine-tuning")
print("-" * 40)
print("""
Fine-tuning needs labeled data in instruction format.
Most common format: instruction-input-output triplets.
""")

# Create a sample fine-tuning dataset
sample_dataset = [
    {
        "instruction": "Classify this student query into a category.",
        "input"      : "How do I fix an index out of range error in Python?",
        "output"     : "DEBUGGING"
    },
    {
        "instruction": "Classify this student query into a category.",
        "input"      : "What is the difference between supervised and unsupervised learning?",
        "output"     : "CONCEPT_QUESTION"
    },
    {
        "instruction": "Classify this student query into a category.",
        "input"      : "Write a function to reverse a linked list.",
        "output"     : "CODE_REQUEST"
    },
    {
        "instruction": "Generate a resume bullet point for this achievement.",
        "input"      : "Built a chatbot using Python and OpenAI API",
        "output"     : "Engineered an AI-powered chatbot using Python and OpenAI API, automating student Q&A and reducing response time by 75%."
    },
    {
        "instruction": "Generate a resume bullet point for this achievement.",
        "input"      : "Created a RAG pipeline with LangChain and ChromaDB",
        "output"     : "Developed a RAG pipeline using LangChain and ChromaDB, enabling semantic search across 500+ documents with 92% retrieval accuracy."
    },
    {
        "instruction": "Explain this AI concept simply.",
        "input"      : "What is overfitting?",
        "output"     : "Overfitting is when a model memorizes training data instead of learning patterns. Like a student who memorizes answers without understanding -- fails on new questions."
    },
]

# Format as Alpaca-style (most common fine-tuning format)
def format_alpaca(sample):
    """Convert to Alpaca instruction format used by most open LLMs."""
    if sample["input"]:
        return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""
    else:
        return f"""### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""

print("  Sample fine-tuning dataset (Alpaca format):\n")
for i, sample in enumerate(sample_dataset[:3], 1):
    formatted = format_alpaca(sample)
    print(f"  --- Sample {i} ---")
    print(formatted)
    print()

print(f"  Total samples in dataset: {len(sample_dataset)}")
print(f"  (Real fine-tuning needs 100-10,000+ samples)")

# Save dataset to JSON
import json
with open("finetune_dataset.json", "w", encoding="utf-8") as f:
    json.dump(sample_dataset, f, indent=2, ensure_ascii=False)
print(f"  Saved to finetune_dataset.json")


# -- SECTION 4: Fine-tuning with HuggingFace PEFT -------------
print("\n\nSECTION 4: Fine-tuning with PEFT/LoRA (Code Walkthrough)")
print("-" * 40)
print("""
This is the code structure for LoRA fine-tuning.
Run this on Google Colab (free T4 GPU) for actual training.

Steps:
  1. Load base model (e.g. LLaMA-2-7B or Mistral-7B)
  2. Apply LoRA config
  3. Prepare dataset
  4. Train
  5. Save adapter
  6. Load and use
""")

lora_code = '''
# ============================================================
# LoRA Fine-tuning Template (run on Google Colab)
# pip install transformers peft datasets bitsandbytes
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

# ── Step 1: Load base model ───────────────────────────────────
model_name = "microsoft/phi-2"  # Small model, free Colab can handle!
# Other options: "mistralai/Mistral-7B-v0.1", "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype = torch.float16,
    device_map  = "auto"   # automatically uses GPU if available
)

# ── Step 2: Apply LoRA config ──────────────────────────────────
lora_config = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = 8,        # rank -- lower = faster, higher = better
    lora_alpha   = 16,       # scaling factor
    lora_dropout = 0.1,      # regularization
    target_modules = ["q_proj", "v_proj"]  # which layers to adapt
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 2,783,506,432 || 0.15%

# ── Step 3: Prepare dataset ────────────────────────────────────
def tokenize(sample):
    text = f"### Instruction:\\n{sample['instruction']}\\n\\n### Input:\\n{sample['input']}\\n\\n### Response:\\n{sample['output']}"
    return tokenizer(text, truncation=True, max_length=512, padding="max_length")

dataset = Dataset.from_list(your_dataset)  # replace with your data
tokenized = dataset.map(tokenize)

# ── Step 4: Train ──────────────────────────────────────────────
from transformers import Trainer

training_args = TrainingArguments(
    output_dir          = "./lora_output",
    num_train_epochs    = 3,
    per_device_train_batch_size = 4,
    learning_rate       = 2e-4,
    fp16                = True,   # mixed precision -- saves memory
    logging_steps       = 10,
    save_steps          = 100,
)

trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = tokenized,
)

trainer.train()

# ── Step 5: Save the LoRA adapter ─────────────────────────────
model.save_pretrained("./my_lora_adapter")
tokenizer.save_pretrained("./my_lora_adapter")

# ── Step 6: Load and use ───────────────────────────────────────
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(model_name)
model      = PeftModel.from_pretrained(base_model, "./my_lora_adapter")

inputs = tokenizer("### Instruction:\\nClassify this query.\\n\\n### Input:\\nHow to fix a bug?\\n\\n### Response:\\n", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
'''

print(lora_code)

# Save the template to a file
with open("lora_finetune_template.py", "w", encoding="utf-8") as f:
    f.write(lora_code)
print("  Saved to lora_finetune_template.py")
print("  Upload this to Google Colab to run with free GPU!")


# -- SECTION 5: Training Metrics Understanding ----------------
print("\n\nSECTION 5: Understanding Training Metrics")
print("-" * 40)
print("""
When training you'll see these metrics -- here's what they mean:

  Loss (training loss)
  --------------------
  -> Measures how wrong the model is
  -> Should DECREASE over time
  -> If not decreasing: learning rate too low, or bad data
  -> If decreasing then increasing: overfitting

  Perplexity
  ----------
  -> How "surprised" the model is by the text
  -> Lower = better
  -> Formula: perplexity = exp(loss)
  -> Good LLM: perplexity < 10

  Learning Rate
  -------------
  -> How big steps the optimizer takes
  -> Too high: training unstable, loss jumps around
  -> Too low: training too slow
  -> Common values: 1e-4 to 5e-4 for LoRA

  Epoch
  -----
  -> One full pass through the entire training dataset
  -> More epochs = better learning but risk of overfitting
  -> For LoRA: usually 1-5 epochs is enough
""")

# Simulate training metrics
print("  Simulated training run (3 epochs, 100 steps each):")
print(f"\n  {'Step':>6} {'Loss':>8} {'Perplexity':>12} {'LR':>10}")
print(f"  {'─'*6} {'─'*8} {'─'*12} {'─'*10}")

np.random.seed(42)
losses = []
for step in range(0, 301, 50):
    # Simulate decreasing loss with noise
    loss = 2.5 * np.exp(-step/150) + 0.3 + np.random.randn() * 0.05
    loss = max(0.2, loss)
    perplexity = np.exp(loss)
    lr   = 2e-4 * (0.95 ** (step // 50))   # lr decay
    losses.append(loss)
    print(f"  {step:>6} {loss:>8.4f} {perplexity:>12.4f} {lr:>10.2e}")

print(f"\n  Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
print(f"  Improvement: {(1 - losses[-1]/losses[0])*100:.1f}% reduction in loss")


# -- SECTION 6: Fine-tuning vs Prompting vs RAG ---------------
print("\n\nSECTION 6: When to Use What -- Decision Framework")
print("-" * 40)

scenarios = [
    {
        "scenario"  : "Company FAQ chatbot with 200 documents",
        "best"      : "RAG",
        "reason"    : "Documents change frequently, need source citations"
    },
    {
        "scenario"  : "AI that writes in your company's exact brand voice",
        "best"      : "Fine-tuning",
        "reason"    : "Style/tone change requires fine-tuning, not just prompting"
    },
    {
        "scenario"  : "Classify support tickets into 5 categories",
        "best"      : "Fine-tuning",
        "reason"    : "Consistent classification with labeled data -- fine-tune wins"
    },
    {
        "scenario"  : "General Q&A assistant for students",
        "best"      : "Prompt Engineering",
        "reason"    : "General knowledge task -- good system prompt is enough"
    },
    {
        "scenario"  : "Code review bot for your codebase",
        "best"      : "RAG + Prompting",
        "reason"    : "Feed codebase as context via RAG, use prompts for review format"
    },
    {
        "scenario"  : "Medical diagnosis from patient records",
        "best"      : "Fine-tuning + RAG",
        "reason"    : "Needs domain expertise (fine-tune) + patient data (RAG)"
    },
]

print(f"\n  {'Scenario':<45} {'Best Approach':<20} {'Reason'}")
print(f"  {'─'*45} {'─'*20} {'─'*40}")
for s in scenarios:
    print(f"  {s['scenario']:<45} {s['best']:<20} {s['reason']}")

print()
print("=" * 60)
print("Script 2 complete! Fine-tuning covered.")
print("Key concepts:")
print("  [OK] Fine-tuning vs RAG vs Prompting")
print("  [OK] LoRA math and intuition")
print("  [OK] Dataset preparation (Alpaca format)")
print("  [OK] PEFT/LoRA code template for Colab")
print("  [OK] Training metrics explained")
print("  [OK] Decision framework: when to use what")
print("=" * 60)
