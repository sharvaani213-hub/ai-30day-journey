
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
    text = f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
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

inputs = tokenizer("### Instruction:\nClassify this query.\n\n### Input:\nHow to fix a bug?\n\n### Response:\n", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
