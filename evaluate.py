import os
import argparse
import json

import torch
import torch.distributed as dist
import transformers
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, HfArgumentParser, Trainer,
                          set_seed, TrainingArguments)
from datasets import load_dataset, concatenate_datasets

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def attach_eval_prompt(entry, prompt_ex):
  if 'question' in entry.keys():
    input_text = prompt_ex + "\n\nFollow the format above and answer the following question in a single number.\n" + f"Question: {entry['question'].strip()}\nAnswer: "
  elif 'problem' in entry.keys():
    input_text = prompt_ex + "\n\nFollow the format above and solve the following problem.\n" + f"Problem: {entry['problem'].strip()}\nSolution: "
  entry["input_text"] = input_text
  return entry

def create_prompt(examples):
  prompt = ""
  for ex in examples:
    if 'question' in ex.keys():
      prompt += f"Question: {ex['question'].strip()}\nAnswer: {ex['answer'].strip()}\n\n"
    elif 'problem' in ex.keys():
      prompt += f"Problem: {ex['problem'].strip()}\nSolution: {ex['solution'].strip()}\n\n"
  return prompt.strip()

def tokenize(entry, tokenizer):
   outputs = tokenizer(
      entry["input_text"],
      truncation=True,
      max_length = 3000,
      return_tensors = 'pt',
   )
   entry["input_ids"] = outputs
   return entry

def load_model(model_name_path, tokenizer, torch_dtype=torch.bfloat16):
  is_peft = os.path.exists(os.path.join(model_name_path, "adapter_config.json"))
  if is_peft:
    config = LoraConfig.from_pretrained(model_name_path)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch_dtype, device_map="auto", cache_dir="/gscratch/xlab/olo126/.cache")
    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
      base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, model_name_path, device_map="auto")
  else:
    model = AutoModelForCausalLM.from_pretrained(model_name_path, torch_dtype=torch_dtype, device_map="auto", cache_dir="/gscratch/xlab/olo126/.cache")
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
      model.resize_token_embeddings(len(tokenizer))

  for name, param in model.named_parameters():
    if 'lora' in name or 'Lora' in name:
      param.requires_grad = True
  return model

def main(args):
  set_seed(2)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  tokenizer = AutoTokenizer.from_pretrained(args.model_path)
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
  model = load_model(args.model_path, tokenizer, torch.bfloat16)
  
  if args.task == "gsm8k":
    gsm8k_test = load_dataset("openai/gsm8k", "main", split="test", cache_dir="/gscratch/xlab/olo126/.cache").shuffle(seed=2)
    few_shot = gsm8k_test.select(range(8))
    test_raw = gsm8k_test.select(range(8, len(gsm8k_test)))
    few_shot_prompt = create_prompt(few_shot)
    test_dataset = test_raw.map(attach_eval_prompt, fn_kwargs={'prompt_ex': few_shot_prompt})
    #test_dataset = test_dataset.map(tokenize, fn_kwargs={'tokenizer': tokenizer})
    print("gsm8k done")
    dataset = test_dataset


  elif args.task == "comp_math":

    math_test = load_dataset("hendrycks/competition_math", split="test", cache_dir="/gscratch/xlab/olo126/.cache", trust_remote_code=True).shuffle(seed=2)
    few_shot = math_test.select(range(8))
    test_raw = math_test.select(range(8, len(math_test)))
    test_raw = math_test.select(range(8, len(math_test)))
    few_shot_prompt = create_prompt(few_shot)
    test_dataset = test_raw.map(attach_eval_prompt, fn_kwargs={'prompt_ex': few_shot_prompt})
    #test_dataset = test_dataset.map(tokenize, fn_kwargs={'tokenizer': tokenizer})
    print("MATH done")
    dataset = test_dataset
  
  print("FEW SHOT PROMPT")
  print(few_shot_prompt)
  prompt_len = len(few_shot_prompt)

  # pass prompts into the model
  step = 15
  gen_list = []
  correct_list = []
  correct = 0
  answers = dataset['answer'] if args.task == 'gsm8k' else dataset['solution']
  for j in range(0,len(dataset['input_text']),step):
    end = min(j+step, len(dataset['input_text']))
    #inputs = dataset['input_ids'][j:end]
    inputs = tokenizer(
      dataset["input_text"][j:end],
      truncation=True,
      padding="longest",
      max_length = 3000,
      return_tensors = 'pt',
    ).to(device)
    gen = model.generate(**inputs, top_p=0.9, temperature=0.1, max_length=inputs['input_ids'].shape[1]+300)
    decoded = tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # postprocess model outputs to get the aswer
    if args.task == 'gsm8k':
      for i in range(end-j):
        print("\nWHOLE GEN\n")
        print(decoded[i])
        trunc = decoded[i][prompt_len:]
        print("\nNO PROMPT\n")
        print(trunc)
        prob = trunc.split("Question:")[1].split("Answer:")[0].strip()
        expln = trunc.split("Answer:")[1].split("####")[0].strip()
        ans = trunc.split("####")[1].strip() if len(trunc.split("####")) > 1 else trunc
        ans = ans.split("\n\n")[0].strip()
        gen_list.append({"Question": prob, "Explanation:": expln, "Answer": ans, "Correct": answers[j+i]})
        print("\nTRUNC-ED GEN\n")
        print(gen_list[-1])
        # check answer
        if answers[j+i].split("####")[1].strip() == ans:
          correct+=1
          correct_list.append(prob + " " + expln + f" #### {ans}.")
        print(correct/(len(gen_list)))
    elif args.task == 'comp_math':
      for i in range(end-j):
        print("\nWHOLE GEN\n")
        print(decoded[i])
        trunc = decoded[i][prompt_len:]
        print("\nNO PROMPT\n")
        print(trunc)
        prob = trunc.split("Problem:")[1].split("Solution:")[0].strip()
        expln = trunc.split("Solution:")[1].split("\\boxed{")[0].strip()
        ans = trunc.split("\\boxed{")[1].split("}")[0].strip()
        clean_ans = remove_boxed(last_boxed_only_string(trunc))
        gt_ans = remove_boxed(last_boxed_only_string(answers[j+1]))
        gen_list.append({"Problem": prob, "Explanation:": expln, "Solution": ans, "Correct": answers[j+i]})
        print("\nTRUNC-ED GEN\n")
        print(gen_list[-1])
        # check answer
        if gt_ans == clean_ans:
          correct+=1
          correct_list.append(prob + " " + expln + f"  {ans}.")
        print(correct/(len(gen_list)))
  # record results
  with open(os.path.join(args.output_dir, f"{args.task}_{args.model_path.split('/')[-1]}.json"), 'w', encoding='utf-8') as f:
    json.dump(gen_list, f, ensure_ascii=False, indent=4)
  with open(os.path.join(args.output_dir, f"{args.task}_score_{args.model_path.split('/')[-1]}.json"), 'w', encoding='utf-8') as f:
    json.dump([correct, correct/len(dataset['input_text']), len(dataset['input_text'])], f, ensure_ascii=False, indent=4)
  


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path')
  parser.add_argument('--task')
  parser.add_argument('--output_dir')
  parser.add_argument('--lora', action="store_false")
  parser.add_argument('--lora_r', default = 128)
  parser.add_argument('--lora_a', default = 512)
  parser.add_argument('--lora_dropout', default = 0.1)
  parser.add_argument('--lora_target_modules', default = ["q_proj", "k_proj", "v_proj", "o_proj"])
  args = parser.parse_args()
  main(args)