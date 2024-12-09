import os
import argparse

import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser, Trainer,
                          set_seed, TrainingArguments)
from datasets import load_dataset, concatenate_datasets


def combine_data(entry):
  if 'question' in entry.keys():
    entry['question'] = entry['question'] + "\n" + entry['answer']
  elif 'problem' in entry.keys():
    entry['problem'] = entry['problem'] + "\n" + entry['solution']
  return entry

def tokenize(entry, tokenizer):
   outputs = tokenizer(
      entry["text"],
      truncation=True,
      max_length = 3000,
   )
   return outputs

def main(args):
  set_seed(2)

  gsm8k_train = load_dataset("openai/gsm8k", "main", split="train", cache_dir="/gscratch/xlab/olo126/.cache").shuffle(seed=2)
  gsm8k_train = gsm8k_train.map(combine_data)
  gsm8k_train = gsm8k_train.rename_column("question", "text")
  gsm8k_train = gsm8k_train.remove_columns("answer")
  print("gsm8k done")

  math_train = load_dataset("hendrycks/competition_math", split="train", cache_dir="/gscratch/xlab/olo126/.cache", trust_remote_code=True).shuffle(seed=2)
  math_train = math_train.map(combine_data)
  math_train = math_train.rename_column("problem", "text")
  math_train = math_train.remove_columns(["solution", "level", "type"])
  print("MATH done")

  owm_train = load_dataset("open-web-math/open-web-math", split="train", cache_dir="/gscratch/xlab/olo126/.cache").shuffle(seed=2)
  #owm_train = owm_train.filter(lambda example, idx: idx < len(owm_train) // 10, with_indices=True).remove_columns(["url", "date", "metadata"])
  print("OpenWebMath done")

  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", padding_side="left", cache_dir="/gscratch/xlab/olo126/.cache")
  model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", cache_dir="/gscratch/xlab/olo126/.cache")

  warmup_gsm8k = gsm8k_train.select(range(len(gsm8k_train) // 10))
  warmup_math = math_train.select(range(len(math_train) // 10))
  warmup_owm = owm_train.select(range(len(owm_train) // 1000))
  warmup_dataset = concatenate_datasets([warmup_gsm8k, warmup_math, warmup_owm]).map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer}).shuffle(seed=2)
  print(warmup_dataset[0])

  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

  embedding_size = model.get_input_embeddings().weight.shape[0]
  if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))
    if isinstance(model, PeftModel):
      model.get_input_embeddings().weight.requires_grad = False
      model.get_output_embeddings().weight.requires_grad = False

  if not isinstance(model, PeftModel) and args.lora:
    lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=args.lora_r,
      lora_alpha=args.lora_a,
      lora_dropout=args.lora_dropout,
      target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # for checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
  

  training_args = TrainingArguments(
     output_dir='results/warmup_results',
     lr_scheduler_type='linear',
     warmup_ratio=0.03,
     save_strategy='epoch',
     num_train_epochs=4,
     bf16=True,
     tf32=False,
     overwrite_output_dir=True,
     report_to='wandb',
     seed=2,
     optim="adamw_torch",
     learning_rate=2e-05,
     per_device_train_batch_size=1,
     gradient_accumulation_steps=32,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=warmup_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(
      tokenizer=tokenizer, mlm=False)
  )

  train_result = trainer.train()
  trainer.save_model()
  metrics = train_result.metrics
  metrics['train_samples'] = len(warmup_dataset)
  trainer.log_metrics("train", metrics)
  trainer.save_metrics("train", metrics)
  trainer.save_state()

  """
  if isinstance(model, PeftModel):
    pytorch_model_path = os.path.join(
       training_args.output_dir, "pytorch_model_fsdp.bin")
    os.remove(pytorch_model_path) if os.path.exists(
       pytorch_model_path) else None
"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--lora', default = True)
  parser.add_argument('--lora_r', default = 128)
  parser.add_argument('--lora_a', default = 512)
  parser.add_argument('--lora_dropout', default = 0.1)
  parser.add_argument('--lora_target_modules', default = ["q_proj", "k_proj", "v_proj", "o_proj"])
  args = parser.parse_args()
  main(args)