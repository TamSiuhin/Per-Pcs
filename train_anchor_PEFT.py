import sys 
sys.path.append("..") 

import argparse
import copy
import datetime
import json
import os
import time
from pathlib import Path
from rank_bm25 import BM25Okapi
from functools import partial

import numpy as np
# import timm.optim.optim_factory as optim_factory
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
from engine_finetuning import train_one_epoch, val_one_epoch, load_model, load_generator_from_raw, load_generator_from_trained
from torch.utils.data import Dataset
# from torch.utils.tensorboard import SummaryWriter
# from util.misc import NativeScalerWithGradNormCount as NativeScaler
from utils import split_batch, get_first_k_tokens, print_trainable_parameters, name2taskid
from utils import extract_citation_title, extract_option, extract_movie, extract_news_cat, extract_news_headline, extract_product_review, extract_scholarly_title, extract_tweet_paraphrasing

from llama import Tokenizer
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
from tqdm import trange, tqdm

def extract_article(text):
    marker = "] description: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

class InstructionDataset(Dataset):
    def __init__(self, data_list, tokenizer_path, max_tokens=2048):
        self.ann = data_list

        self.max_words = max_tokens
        tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")
        self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        prompt = ann['prompt']
        example = ann['full_prompt']

        prompt = torch.tensor(self.tokenizer1.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer1.encode(example, bos=True, eos=True), dtype=torch.int64)

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        return example, labels, prompt


def collate_fn(batch, max_length=2048):
    examples, labels, prompts = zip(*batch)
    # Trim sequences to max_length
    trimmed_examples = [example[:max_length] for example in examples]
    trimmed_labels = [label[:max_length] for label in labels]
    
    # Determine the maximum sequence length after trimming but capped at max_length
    max_length = min(max([len(example) for example in trimmed_examples]), max_length)

    # Pad sequences to the determined max_length
    padded_examples = torch.stack([torch.cat((example, torch.zeros(max_length - len(example), dtype=torch.int64) - 1)) if len(example) < max_length else example for example in trimmed_examples])
    padded_labels = torch.stack([torch.cat((label, torch.zeros(max_length - len(label), dtype=torch.int64) - 1)) if len(label) < max_length else label for label in trimmed_labels])

    example_masks = padded_examples.ge(0)
    label_masks = padded_labels.ge(0)

    padded_examples[~example_masks] = 0
    padded_labels[~label_masks] = 0

    example_masks = example_masks.float()
    label_masks = label_masks.float()

    return padded_examples, padded_labels, example_masks


def get_args_parser():
    parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=6,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--max_step", default=100, type=int)

    parser.add_argument("--warmup_epochs", default=0, type=int)

    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument("--llama_model_path", default="/afs/crc.nd.edu/user/z/ztan3/.llama/checkpoints/Llama-2-7b", type=str, help="path of llama model")
    parser.add_argument("--tokenizer_path", default="/afs/crc.nd.edu/user/z/ztan3/.llama/checkpoints/Llama-2-7b", type=str, help="path of llama model tokenizer")

    parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")

    parser.add_argument("--model", default="llama7B_lora", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--max_seq_len", type=int, default=3000, metavar="LENGTH", help="the maximum sequence length")
    
    parser.add_argument("--w_lora", type=bool, default=True, help="use lora or not")

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (default: 0.05)")

    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate (absolute lr)")
    parser.add_argument("--clip", type=float, default=0.3, help="gradient clipping")

    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0"
    )


    # Dataset parameters
    # parser.add_argument("--test_data_path", default="./data/movie_tagging/user_anchor_candidate.json", type=str, help="dataset path")
    # parser.add_argument("--train_data_path", default="/afs/crc.nd.edu/user/z/ztan3/Private/LoRA-composition/LaMP_data-final/movie/user_base_LLM.json", type=str, help="dataset path")
    
    parser.add_argument("--output_dir", default="./output/movie_tagging/Anchor_PEFT/LoRA", help="path where to save, empty for no saving")

    parser.add_argument("--log_dir", default="./output", help="path where to tensorboard log")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--lora_ckpt", default='./output/movie_tagging/task-base_LLM/lora_ckpt.pt', help="resume lora from checkpoint")
    parser.add_argument("--grad_ckpt", type=bool, default=True, help="whether to user gradient checkpoint, recommend TRUE!!")

    parser.add_argument("--anchor_path", default='./anchor_selection/history_avg/anchor_user_idx.pt', help="resume lora from checkpoint")

    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # generation hyperparameters
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p")
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature")
    parser.add_argument("--max_gen_len", type=int, default=100, help="top_p")

    parser.add_argument("--k_list", type=list, default=[1,2,4], help="top_p")
    parser.add_argument('--infer', default=False, action=argparse.BooleanOptionalAction)

    return parser


args = get_args_parser()
args = args.parse_args()
# args.test_data_path = f"./data/{args.task_name}/test_100/user_test_100.json"
args.train_data_path = f"./data/{args.task_name}/user_base_LLM.json"
args.test_data_path = f"./data/{args.task_name}/user_anchor_candidate.json"

with open(f'./data/{args.task_name}/profile-id2text.json', 'r') as f:
    all_profile = json.load(f)

import random

def process_train_data(user, k_list=[1,2,4]):

    train_data = []

    format_flag = False
    if args.task_name == "movie_tagging":
        extract_article = extract_movie
        format_flag = True
    elif args.task_name == "news_categorize":
        extract_article = extract_news_cat
        format_flag = True
    elif args.task_name == "news_headline":
        extract_article = extract_news_headline
        format_flag = True
    elif args.task_name == "product_rating":
        extract_article = extrat_product_review
        format_flag = True
    elif args.task_name == "scholarly_title":
        extract_article = extract_scholarly_title
        format_flag = True
    elif args.task_name == "tweet_paraphrase":
        extract_article = extrat_tweet_paraphrasing
    
    with open('./prompt/prompt.json', 'r') as f:
        prompt_template = json.load(f)

    user_profile = all_profile[str(user['user_id'])]
    
    for idx, q in enumerate(user['profile']):
        for key, value in q.items():
            q[key] = get_first_k_tokens(q[key], 768)

        prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)
        
        if idx != 0 and format_flag==True:
            # k = random.sample([1,2,4], 1)[0]
            # for k in k_list:
            k = 1
            visible_history_list = user['profile'][:idx]
            for p in visible_history_list:
                for key, value in p.items():
                    p[key] = get_first_k_tokens(p[key], 368)

            history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]
            tokenized_corpus = [doc.split(" ") for doc in history_list]
            bm25 = BM25Okapi(tokenized_corpus)
            
            tokenized_query = prompt_template[args.task_name]["retrieval_query"].format(**q).split(' ')
            retrieved_history = bm25.get_top_n(tokenized_query, history_list, n=k)

            history_string = "".join(retrieved_history)

            if format_flag:
                prompt = f"### User History:\n{history_string}\n\n" + prompt
                full_prompt = f"### User History:\n{history_string}\n\n" + full_prompt
                
                prompt = f"### User Profile:\n{user_profile}\n\n" + prompt
                full_prompt = f"### User Profile:\n{user_profile}\n\n" + full_prompt
                    
            train_data.append(
                {
                    "prompt": prompt,
                    "full_prompt": full_prompt
                }
            )
    
    for idx, q in enumerate(user['profile']):

        for key, value in q.items():
            q[key] = get_first_k_tokens(q[key], 768)

        prompt = prompt_template[args.task_name]['OPPU_input'].format(**q)
        full_prompt = prompt_template[args.task_name]['OPPU_full'].format(**q)

        train_data.append(
            {
                "prompt": prompt,
                "full_prompt": full_prompt
            }
        )

    return train_data


def main(args):
    torch.set_default_device('cuda')

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed # + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    anchor_idx = torch.load(args.anchor_path)

    with open(args.test_data_path, 'r') as f:
        all_user_data = json.load(f)
    
    # define the model
    model = load_model(
        ckpt_dir=args.llama_model_path,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.batch_size,
        lora_path=args.lora_ckpt,
        w_lora=args.w_lora,
        grad_ckpt=args.grad_ckpt
    )

    model.to(device)
    model.print_trainable_params()
    model.merge_lora_parameters()
    print('merged!!')


    # print("Model = %s" % str(model))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)

    pred_all = [[] for _ in range(len(args.k_list)+1)]

    for idx in tqdm(range(len(anchor_idx))):
        idx_all_test = anchor_idx[idx]
        user = all_user_data[idx_all_test]

        user_out_dir = os.path.join(args.output_dir, 'user_{}'.format(user['user_id']))

        Path(user_out_dir).mkdir(parents=True, exist_ok=True)

        model.reset_lora_parameters()
        model.set_lora_trainable()
        
        data_list = process_train_data(user)

        dataset_train = InstructionDataset(
            data_list=data_list, tokenizer_path=args.tokenizer_path, max_tokens=args.max_seq_len
        )
    
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)

        # os.makedirs(args.log_dir, exist_ok=True)
        # log_writer = SummaryWriter(log_dir=args.log_dir)
        # else:
        log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            # sampler=sampler_train,
            shuffle=True,
            batch_size=args.batch_size,
            # num_workers=args.num_workers,
            # pin_memory=args.pin_mem,
            drop_last=False,
            generator=torch.Generator(device='cuda'),
            collate_fn=partial(collate_fn, max_length=args.max_seq_len),
        )

        # following timm: set wd as 0 for bias and norm layers
        # param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
        # loss_scaler = NativeScaler()


        args.cur_step = 0
        try:
            print(f"Start training for {args.max_step} steps")
        except:
            print(f"Start training for {args.epochs} epochs")        
        
        start_time = time.time()
        # for epoch in range(args.epochs):
        epoch = 0

        while args.cur_step < args.max_step:
            train_stats = train_one_epoch(
                model, data_loader_train, optimizer, device, epoch, None, log_writer=log_writer, args=args
            )
            epoch += 1

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                # **{f"val_{k}": v for k, v in val_stats.items()},
            }

            if args.output_dir:
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(user_out_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")

        torch.save(model.lora_state_dict(), os.path.join(user_out_dir, 'lora_ckpt.pt'))
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    
if __name__ == "__main__":

    # args = get_args_parser()
    # args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
