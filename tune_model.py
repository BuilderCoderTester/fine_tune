import os 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig

# Load dataset
dataset = load_dataset("your_dataset_name")

# Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load tokenizer
model_name_or_path = "meta-llama/Llama-2-7b-hf"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
max_length = 512

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to base model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=200,
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    evaluation_strategy="steps",
    eval_steps=200,
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"] if "test" in dataset else None,
    peft_config=lora_config,
    dataset_text_field="text",  # Change to your text field name
    max_seq_length=max_length,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
print("Saving model...")
trainer.model.save_pretrained("./lora_finetuned_model/final")
tokenizer.save_pretrained("./lora_finetuned_model/final")

print("Training complete!")

# ============================================
# INFERENCE
# ============================================

print("\n" + "="*50)
print("Starting Inference...")
print("="*50 + "\n")

# Load the fine-tuned model for inference
inference_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load the LoRA adapter
inference_model = PeftModel.from_pretrained(
    inference_model, 
    "./lora_finetuned_model/final"
)
inference_model.eval()

# Test prompts
test_prompts = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot.",
]

print("Generating predictions...\n")

for i, prompt in enumerate(test_prompts, 1):
    print(f"Prompt {i}: {prompt}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output
    with torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and print
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}\n")
    print("-" * 50 + "\n")

print("Inference complete!")