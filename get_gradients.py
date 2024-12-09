import argparse
import os
from copy import deepcopy
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import normalize
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets
from trak.projectors import BasicProjector, CudaProjector, ProjectionType
from tqdm import tqdm

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

def load_model(model_name_path, tokenizer, torch_dtype=torch.bfloat16):
  is_peft = os.path.exists(os.path.join(model_name_path, "adapter_config.json"))
  if is_peft:
    config = LoraConfig.from_pretrained(model_name_path)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto")
    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
      base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_name_path, device_map="auto")
  else:
    model = AutoModelForCausalLM.from_pretrained(
       model_name_path, torch_dtype=torch_dtype, device_map="auto")

  for name, param in model.named_parameters():
    if 'lora' in name or 'Lora' in name:
      param.requires_grad = True
  return model

def _project(current_full_grads, projected_grads, projectors, model_id, proj_dim):
  current_full_grads = torch.stack(current_full_grads).to(torch.float16)
  for i, projector in enumerate(projectors):
    current_projected_grads = projector.project(
      current_full_grads, model_id=model_id)
    projected_grads[proj_dim[i]].append(current_projected_grads.cpu())

def _save(projected_grads, output_dirs, proj_dim, count):
  for dim in proj_dim:
    if len(projected_grads[dim]) == 0:
      continue
    projected_grads[dim] = torch.cat(projected_grads[dim])
    output_dir = output_dirs[dim]
    outfile = os.path.join(output_dir, f"grads-{count}.pt")
    torch.save(projected_grads[dim], outfile)
    print(
      f"Saving {outfile}, {projected_grads[dim].shape}", flush=True)
    projected_grads[dim] = []

def get_max_saved_index(output_dir: str, prefix="reps") -> int:
  files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]
  index = [int(file.split(".")[0].split("-")[1]) for file in files]
  return max(index) if len(index) > 0 else -1

def obtain_gradients_with_adam(model, batch, avg, avg_sq):
  """ obtain gradients with adam optimizer states. """
  beta1 = 0.9
  beta2 = 0.999
  eps = 1e-08

  loss = model(**batch).loss
  loss.backward()

  vectorized_grads = torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])

  updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
  updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
  vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)

  return vectorized_grads

def merge_and_normalize_info(output_dir: str, prefix="reps"):
  """ Merge and normalize the representations and gradients into a single file. """
  info = os.listdir(output_dir)
  info = [file for file in info if file.startswith(prefix)]
  # Sort the files in ascending order
  info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
  merged_data = []
  for file in info:
    data = torch.load(os.path.join(output_dir, file))
    normalized_data = normalize(data, dim=1)
    merged_data.append(normalized_data)
  merged_data = torch.cat(merged_data, dim=0)

  output_file = os.path.join(output_dir, f"all_orig.pt")
  torch.save(merged_data, output_file)
  print(
    f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")
  
def merge_info(output_dir: str, prefix="reps"):
  """ Merge the representations and gradients into a single file without normalization. """
  info = os.listdir(output_dir)
  info = [file for file in info if file.startswith(prefix)]
  # Sort the files in ascending order
  info.sort(key=lambda x: int(x.split(".")[0].split("-")[1]))
  merged_data = []
  for file in info:
    data = torch.load(os.path.join(output_dir, file))
    merged_data.append(data)
  merged_data = torch.cat(merged_data, dim=0)

  output_file = os.path.join(output_dir, f"all_unormalized.pt")
  torch.save(merged_data, output_file)
  print(f"Saving the unnormalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")

def main(args):
  set_seed(2)

  tokenizer = AutoTokenizer.from_pretrained(args.model_path)
  model = load_model(args.model_path, tokenizer, torch.bfloat16)

  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
  
  if args.initialize_lora:
    assert not isinstance(model, PeftModel)
    lora_config = LoraConfig(
      task_type=TaskType.CAUSAL_LM,
      inference_mode=False,
      r=args.lora_r,
      lora_alpha=args.lora_a,
      lora_dropout=args.lora_dropout,
      target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, lora_config)

  if isinstance(model, PeftModel):
    model.print_trainable_parameters()
  
  adam_optimizer_state = None
  if args.info_type == "grads" and args.gradient_type == "adam":
    optimizer_path = os.path.join(args.model_path, "optimizer.pt")
    adam_optimizer_state = torch.load(optimizer_path, map_location="cpu")["state"]
  
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

  sub_gsm8k = gsm8k_train.select(range(len(gsm8k_train) // 10, len(gsm8k_train)))
  sub_math = math_train.select(range(len(math_train) // 10, len(math_train)))
  sub_owm = owm_train.select(range(len(owm_train) // 1000, len(owm_train) // 1000 + len(owm_train) // 20))
  pretrain_dataset = concatenate_datasets([sub_gsm8k, sub_math, sub_owm]).map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer}).shuffle(seed=2)

  print(pretrain_dataset[0])
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
  dataloader = DataLoader(pretrain_dataset, batch_size=1, collate_fn=data_collator)

  model_id = 0  # model_id is used to draft the random seed for the projectors
  block_size = 128  # fixed block size for the projectors
  projector_batch_size = 16  # batch size for the projectors
  torch.random.manual_seed(0)  # set the random seed for torch

  project_interval = 16  # project every 16 batches
  save_interval = 160  # save every 160 batches

  device = next(model.parameters()).device
  dtype = next(model.parameters()).dtype

  if args.gradient_type == "adam":
    assert adam_optimizer_state is not None
    # first and second moment estimates
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([adam_optimizer_state[n]["exp_avg"].view(-1) for n in range(len(names))])
    avg_sq = torch.cat([adam_optimizer_state[n]["exp_avg_sq"].view(-1) for n in range(len(names))])
    m = avg.to(device)
    v = avg_sq.to(device)

  try:
    num_sms = torch.cuda.get_device_properties(
      device.index).multi_processor_count
    import fast_jl

    # test run to catch at init time if projection goes through
    fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
    projector = CudaProjector
    print("Using CudaProjector")
  except:
    projector = BasicProjector
    print("Using BasicProjector")

  if isinstance(model, PeftModel):
    names = [n for n, p in model.named_parameters(
    ) if p.requires_grad and "lora" not in n]
    assert len(names) == 0
  num_params = sum([p.numel()
                    for p in model.parameters() if p.requires_grad])
  print(f"Total number of parameters that require gradients: {num_params}")

  projectors = []
  for dim in args.gradient_projection_dimension:
    proj = projector(grad_dim=num_params,
                     proj_dim=dim,
                     seed=0,
                     proj_type=ProjectionType.rademacher,
                     device=device,
                     dtype=dtype,
                     block_size=block_size,
                     max_batch_size=projector_batch_size)
    projectors.append(proj)

  count = 0

  # set up a output directory for each dimension
  output_dirs = {}
  for dim in args.gradient_projection_dimension:
    output_dir_per_dim = os.path.join(args.output_dir, f"dim{dim}")
    output_dirs[dim] = output_dir_per_dim
    os.makedirs(output_dir_per_dim, exist_ok=True)
  
  # max index for each dimension
  max_index = min(get_max_saved_index(output_dirs[dim], "grads") for dim in args.gradient_projection_dimension)

  # projected_gradients
  full_grads = []  # full gradients
  projected_grads = {dim: [] for dim in args.gradient_projection_dimension}  # projected gradients

  for batch in tqdm(dataloader, total=len(dataloader)):
    for key in batch:
      batch[key] = batch[key].to(device)
    count += 1

    if count <= max_index:
      print("skipping count", count)
      continue

    if args.gradient_type == "adam":
      if count == 1:
        print("Using Adam gradients")
      vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
    """
    elif gradient_type == "sign":
      if count == 1:
        print("Using Sign gradients")
      vectorized_grads = obtain_sign_gradients(model, batch)
    else:
      if count == 1:
        print("Using SGD gradients")
      vectorized_grads = obtain_gradients(model, batch)
    """

    # add the gradients to the full_grads
    full_grads.append(vectorized_grads)
    model.zero_grad()

    if count % project_interval == 0:
      _project(full_grads, projected_grads, projectors, model_id, args.gradient_projection_dimension)
      full_grads = []

    if count % save_interval == 0:
      _save(projected_grads, output_dirs, args.gradient_projection_dimension, count)

    if args.max_samples is not None and count == args.max_samples:
      break

  if len(full_grads) > 0:
    _project(full_grads, projected_grads, projectors, model_id, args.gradient_projection_dimension)
    full_grads = []

  for dim in args.gradient_projection_dimension:
    _save(projected_grads, output_dirs, args.gradient_projection_dimension, count)

  torch.cuda.empty_cache()
  for dim in args.gradient_projection_dimension:
    output_dir = output_dirs[dim]
    merge_and_normalize_info(output_dir, prefix="grads")
    merge_info(output_dir, prefix="grads")

  print("Finished")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path')
  parser.add_argument('--info_type', default = "grads")
  parser.add_argument('--gradient_type', default = "adam")
  parser.add_argument('--gradient_projection_dimension', default=[8192])
  parser.add_argument('--output_dir')
  parser.add_argument('--max_samples', default=None)
  parser.add_argument('--initialize_lora', default=False)
  parser.add_argument('--lora', default = True)
  parser.add_argument('--lora_r', default = 8)
  parser.add_argument('--lora_a', default = 32)
  parser.add_argument('--lora_dropout', default = 0.1)
  parser.add_argument('--lora_target_modules', default = ["q_proj", "k_proj", "v_proj", "o_proj"])
  args = parser.parse_args()
  main(args)