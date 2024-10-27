import math
import sys
from typing import Iterable

import torch
import util.lr_sched as lr_sched
import util.misc as misc
import torch.utils.checkpoint
from llama import Transformer, LLaMA


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):      
        examples = examples.cuda()
        labels = labels.cuda()
        
        # print(examples.size())

        c_loss = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss.backward()
        

        # loss_scaler(loss, optimizer, clip_grad=args.clip, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        
        # misc.all_reduce_mean(loss_value)
        # c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
        
        if getattr(args, 'max_step', None) is not None:
            args.cur_step += 1
            if args.cur_step+1 >= args.max_step:
                break

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_interleave(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, ((qa_examples, qa_labels, qa_example_mask), (user_examples, user_labels, user_example_mask)) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):      
        qa_examples = qa_examples.cuda()
        qa_labels = qa_labels.cuda()
        
        # print(examples.size())

        c_loss = model(qa_examples, qa_labels)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss.backward()

        # loss_scaler(loss, optimizer, clip_grad=args.clip, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # metric_logger.update(closs=c_loss_value)

        # lr = optimizer.param_groups[0]["lr"]
        # metric_logger.update(lr=lr)
        
        user_examples = user_examples.cuda()
        user_labels = user_labels.cuda()
        
        # print(examples.size())

        c_loss = model(user_examples, user_labels)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss.backward()

        # loss_scaler(loss, optimizer, clip_grad=args.clip, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        # torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        # print(metric_logger)


        # misc.all_reduce_mean(loss_value)
        # c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)
        
        if getattr(args, 'max_step', None) is not None:
            args.cur_step += 1
            if args.cur_step+1 >= args.max_step:
                break

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def val_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        with torch.no_grad():
            c_loss = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("c_train_loss", c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



import json
import torch
from llama import ModelArgs, Tokenizer, Transformer
import time
from pathlib import Path



def load_model(
    ckpt_dir: str,
    tokenizer_path: str,
    # local_rank: int,
    # world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    # quantizer: bool=False,
    lora_path: str = None,
    w_lora: bool = False,
    grad_ckpt: bool = True,
    w_gate: bool  =False,
    target_modules = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
):
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # print(checkpoints)
    ckpt_path = checkpoints[0] #[local_rank]

    with open(ckpt_dir + "/params.json", "r") as f:
        params = json.loads(f.read())

    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if lora_path is not None:
        adapter_checkpoint = torch.load(lora_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, w_lora=w_lora, grad_ckpt=grad_ckpt, w_gate=w_gate, target_modules=target_modules, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_args)
    model.eval()
    model.train(False)

    model.load_state_dict(checkpoint, strict=False)

    if lora_path is not None:
        model.load_state_dict(adapter_checkpoint, strict=False)
    
    # torch.set_default_tensor_type(torch.FloatTensor)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    model = model.cuda()
    if w_lora:
        model.set_lora_trainable()
    # if w_lora and lora_path is None:
        # model.set_lora_trainable()
    # elif w_lora and lora_path is not None:
    #     model.eval()
    #     model.train(False)

    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return model


def load_generator_from_raw(
    ckpt_dir: str,
    tokenizer_path: str,
    # local_rank: int,
    # world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    # quantizer: bool=False,
    adapter_path: str = None,
    w_lora: bool = False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[0] #[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if adapter_path is not None:
        adapter_checkpoint = torch.load(adapter_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, w_lora=w_lora, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_args)
    model.eval()
    model.train(False)

    model.load_state_dict(checkpoint, strict=False)

    if adapter_path is not None:
        model.load_state_dict(adapter_checkpoint, strict=False)
    
    torch.set_default_tensor_type(torch.FloatTensor)
    model = model.cuda()
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def load_generator_from_trained(
    model,
    tokenizer_path: str,
) -> LLaMA:
    
    tokenizer = Tokenizer(model_path=tokenizer_path + "/tokenizer.model")
    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model.eval()
    model.train(False)
    
    # torch.set_default_tensor_type(torch.FloatTensor)
    model = model.cuda()
    generator = LLaMA(model, tokenizer)
    return generator