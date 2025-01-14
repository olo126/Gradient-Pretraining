import argparse
import os
import torch
import json
from datasets import load_dataset


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')
    argparser.add_argument('--hg_dataset', action="store_false",
                           help='Whether the train dataset is loaded from hf')

    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    assert len(args.train_file_names) == len(args.train_files)
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    for target_task in args.target_task_names:
        print(f"Target Task {target_task}")
        output_path = os.path.join(args.output_path, target_task)

        score_paths = [os.path.join(
            output_path, f"{task_name}_influence_score.pt") for task_name in args.train_file_names]
        num_samples = []
        for score_path in score_paths:
            num_samples.append(
                len(torch.load(score_path, map_location=device)))
        cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)

        total_samples = sum(num_samples)
        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
        else:
            data_amount_name = f"num{args.max_samples}"

        print("getting all scores")

        all_scores = []
        for score_path, train_file in zip(score_paths, args.train_files):
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
        all_scores = torch.cat(all_scores, dim=0)

        print("sorting scores")
        # sort the scores and output the corresponding data index
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
                              * i for i, line_num in enumerate(num_samples)]).to(device)
        sorted_scores, sorted_index = torch.sort(
            all_scores, dim=0, descending=True)
        sorted_score_file = os.path.join(output_path, f"sorted.csv")

        data_from = data_from[sorted_index]
        sorted_index = file_specific_index[sorted_index]
        
        print("making sorted_score_file")
        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index, name in zip(sorted_scores, sorted_index, data_from):
                    #print(f"writing {sorted_score_file}")
                    file.write(
                        f"{args.train_file_names[name.item()]}, {index.item()}, {round(score.item(), 6)}\n")

        print("doing topk scores")
        topk_scores, topk_indices = torch.topk(
            all_scores.float(), args.max_samples, dim=0, largest=True)

        print("working on all_lines")
        all_lines = []
        for i, train_file in enumerate(args.train_files):
            if args.hg_dataset:
                lines = []
                print("loading owm")
                owm_all = load_dataset("open-web-math/open-web-math", split="train", cache_dir="/gscratch/xlab/olo126/.cache").shuffle(seed=2)
                owm = owm_all.select(range(len(owm_all) // 1000, len(owm_all) // 1000 + len(owm_all) // 20)).shuffle(seed=2)
                print("finished loading owm")
                """
                for i in range(num_samples[i]):
                    lines.append(owm[i])
                    print("appended to lines")
                    print(i)
                print("finished getting all_lines")
                all_lines.append(lines)
                """
                all_lines.append(owm[:num_samples[i]])
            else:
                with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
                    all_lines.append(file.readlines()[:num_samples[i]])

        final_index_list = sorted_index[:args.max_samples].tolist()
        print("finished getting indexes")
        final_data_from = data_from[:args.max_samples].tolist()
        print("finished getting data_from")
        with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
            print("opened ouput file")
            for index, data_from in zip(final_index_list, final_data_from):
                try:
                    #if args.hg_dataset:
                    print("writing output")
                    print(data_from)
                    print(index)
                    print(len(all_lines[data_from]))
                    print(type(all_lines[data_from]))
                    entry = {}
                    for key in all_lines[data_from].keys():
                        entry[key] = all_lines[data_from][key][index]
                    print(entry)
                    file.write(json.dumps(entry))
                except:
                    import pdb
                    pdb.set_trace()
        print(f"Finished Target Task {target_task}")