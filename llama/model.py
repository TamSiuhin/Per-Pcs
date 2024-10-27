# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.checkpoint
from tqdm import trange
import random
from collections import defaultdict
import json
import os
import time


# class EarlyReturnException(Exception):
#     def __init__(self, output):
#         super().__init__()
#         self.output = output

class LoraInjectedLinear(nn.Module):
    def __init__(
        self, in_features, out_features, lora_r, lora_alpha, lora_dropout=0.05, w_gate=False
    ):
        super().__init__()

        if lora_r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {lora_r} must be less or equal than {min(in_features, out_features)}"
            )
        self.lora_r = lora_r
        self.lora_down = nn.Linear(in_features, lora_r, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
        self.lora_up = nn.Linear(lora_r, out_features, bias=False)
        self.scale = 1. * lora_alpha / lora_r

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.w_gate = w_gate
        self.name = None
        self.return_flag = False
        self.corpus = None
        self.label = None
        self.topk = None
        self.temperature = None
        self.sample = False
        self.sample_topk = None
        self.sample_temperature = None
        self.sample_top_p = None
        # self.gate_layer_norm = nn.LayerNorm(
        #     self.params.dim, elementwise_affine=False
        # )
        # self.input_layer_norm = nn.LayerNorm(
        #     self.params.dim, elementwise_affine=False
        # )

        if w_gate:
            self.input_gate = nn.Parameter(torch.zeros(in_features, 1))
            # torch.nn.init.kaiming_uniform_(self.input_gate, nonlinearity='sigmoid')
            # torch.nn.init.xavier_normal_(self.input_gate)
            # torch.nn.init.xavier_uniform_(self.input_ga   te)

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        x = x.to(self.lora_up.weight.dtype)

        if self.return_flag:

            dim = x.size(-1)
            normalized_shape = (dim,)

            # hidden_state = x
            # gate_state = self.corpus['gate'].cuda().squeeze()

            hidden_state = torch.nn.functional.layer_norm(x, normalized_shape=normalized_shape) # (batch_size, seq_len, dim)
            gate_state = torch.nn.functional.layer_norm(self.corpus['gate'].cuda().squeeze(), normalized_shape=normalized_shape) # (100, dim)

            scores = torch.matmul(hidden_state, gate_state.T)
            scores = scores * math.sqrt(1 / dim) # (batch_size, seq_len, 100)
            
            scores = scores[:, :-1, :] # (batch_size, seq_len, 100)
            label = self.label[:, 1:] # (batch_size, seq_len)
            
            # print(label)

            label_mask = label.gt(0) # (batch_size, seq_len)
            scores[~label_mask] = 0 # (batch_size, seq_len, 100)

            # print(scores)
            # print(scores.sum(dim=1))
            # print(scores.size())
    
            summed = torch.sum(scores, dim=1) # (batch_size, 100)
            # print(label[label_mask])
            # print(scores[label_mask].mean(dim=-1))

            count = label_mask.unsqueeze(-1).sum(1)
            # print(count)
            mean_pooled = summed / count # (batch_size, 100)
            # print(mean_pooled)
            scores = mean_pooled.mean(dim=0) # (100)
            
            # scores = scores.mean(dim=1).mean(dim=0)
            # print(scores)

            if self.sample:
                if self.sample_top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float()/self.sample_temperature, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.sample_top_p
                    # Shift the indices to the right to keep at least one token
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    sorted_indices_to_remove[:self.topk] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scores[indices_to_remove] = torch.tensor(float('-inf'))

                    sample_prob = torch.softmax(scores.float()/self.temperature, dim=0)
                    idx = torch.multinomial(sample_prob, num_samples=self.topk, replacement=False)

                    weights_topk_idx = idx
                    weights_topk_value = scores[weights_topk_idx]      

                else:
                    sample_score, candid_idx = torch.topk(scores, k = self.sample_topk)
                    sample_prob = torch.softmax(sample_score.float()/self.sample_temperature, dim=0)
                    idx = torch.multinomial(sample_prob, num_samples=self.topk, replacement=False)

                    weights_topk_idx = candid_idx[idx]
                    weights_topk_value = scores[weights_topk_idx]
            else:
                if self.sample_top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
                    # print(sorted_logits)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits.float()/self.sample_temperature, dim=-1), dim=-1)
                    # print(cumulative_probs)
                    # Remove tokens with cumulative probability above the threshold p
                    sorted_indices_to_remove = cumulative_probs > self.sample_top_p
                    # Shift the indices to the right to keep at least one token
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0

                    sorted_indices_to_keep = ~sorted_indices_to_remove

                    indices_to_keep = sorted_indices[sorted_indices_to_keep]
                    selected_scores = scores[indices_to_keep]

                    weights_topk_idx = indices_to_keep
                    weights_topk_value = selected_scores

                    # print(weights_topk_idx)

                else:
                    weights_topk_value, weights_topk_idx = torch.topk(scores, k=self.topk)

                    # size (batch_size, topk)
                    
            weights_sfm = torch.softmax(weights_topk_value/self.temperature, dim=0)
            # weights_sfm = torch.ones(weights_topk_value.size())* (1/self.topk)


            self.record = {
                "weights": weights_sfm.tolist(),
                "idx": weights_topk_idx.tolist()
            }
            # print(self.record)

            all_lora = self.corpus['loras']
            keys = all_lora[0].keys()
            final_state = {}

            for j in range(weights_topk_idx.size(0)):
                lora = all_lora[weights_topk_idx[j]]

                if j == 0:
                    for key in keys:
                        final_state[key] = weights_sfm[j] * lora[key].cuda()
                else:
                    for key in keys:
                        final_state[key] = final_state[key] + weights_sfm[j] * lora[key].cuda()

            for key, data in final_state.items():
                if 'lora_up' in key:
                    self.lora_up.weight.data = data
                elif 'lora_down' in key:
                    self.lora_down.weight.data = data
            
            self.return_flag = False

        if self.w_gate:
            input_gate_scores = torch.sum(
                x * (self.input_gate.squeeze()), dim=-1
            )
            input_gate_probs = torch.sigmoid(input_gate_scores)
            # print(input_gate_probs)
            x = x * input_gate_probs.unsqueeze(-1)

        result = self.lora_up(self.lora_down(self.dropout(x))) * self.scale
        result = result.to(previous_dtype)
        # print(result)
        return result
    
    def re_init_param(self):
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 4
    max_seq_len: int = 4096

    w_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: Tuple[str] = ('q_proj', 'k_proj', 'v_proj', 'o_proj')     # Option

    grad_ckpt: bool = True
    w_gate: bool = False


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads 
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False,)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False,)
        self.args = args

        if args.w_lora:
            if 'q_proj' in args.target_modules:
                self.lora_wq = LoraInjectedLinear(self.wq.in_features, self.wq.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)
            if 'k_proj' in args.target_modules:
                self.lora_wk = LoraInjectedLinear(self.wk.in_features, self.wk.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)

            if 'v_proj' in args.target_modules:
                self.lora_wv = LoraInjectedLinear(self.wv.in_features, self.wv.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)

            if 'o_proj' in args.target_modules:
                self.lora_wo = LoraInjectedLinear(self.wo.in_features, self.wo.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)

        # if not self.training:
        #     self.cache_k = torch.zeros(
        #         (
        #             args.max_batch_size,
        #             args.max_seq_len,
        #             self.n_local_kv_heads,
        #             self.head_dim,
        #         )
        #     ).cuda()
        #     self.cache_v = torch.zeros(
        #         (
        #             args.max_batch_size,
        #             args.max_seq_len,
        #             self.n_local_kv_heads,
        #             self.head_dim,
        #         )
        #     ).cuda()

    def train(self, mode: bool = True):
        if mode:
            self.cache_k = None
            self.cache_v = None
        else:
            self.cache_k = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
            self.cache_v = torch.zeros(
                (self.args.max_batch_size, self.args.max_seq_len, self.n_local_heads, self.head_dim)
            ).cuda()
        return super().train(mode)


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor. - (batch_size, seq_len, dim)
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        if self.args.w_lora:
            if 'q_proj' in self.args.target_modules:
                xq = xq + self.lora_wq(x)
            if 'k_proj' in self.args.target_modules:
                xk = xk + self.lora_wk(x)
            if 'v_proj' in self.args.target_modules:
                xv = xv + self.lora_wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        if not self.training:
            # print('cache')
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        else:
            assert start_pos==0
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        # print(scores.size())
        # print(scores.dtype)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        if self.args.w_lora and 'o_proj' in self.args.target_modules:
            return self.wo(output) + self.lora_wo(output)
        else:
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        args: ModelArgs
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # self.w1 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )
        # self.w2 = RowParallelLinear(
        #     hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        # )
        # self.w3 = ColumnParallelLinear(
        #     dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        # )

        self.w1 = nn.Linear(dim, hidden_dim, bias=False,)    # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False,)    # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False,)    # up_proj

        self.params = args

        if self.params.w_lora:
            if 'up_proj' in args.target_modules:
                self.lora_w3 = LoraInjectedLinear(self.w3.in_features, self.w3.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)
            if 'down_proj' in args.target_modules:
                self.lora_w2 = LoraInjectedLinear(self.w2.in_features, self.w2.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)
            if 'gate_proj' in args.target_modules:
                self.lora_w1 = LoraInjectedLinear(self.w1.in_features, self.w1.out_features, lora_r=args.lora_r,
                                                  lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, w_gate=args.w_gate)
                
    def forward(self, x):
        up_x = self.w3(x)
        gate_x = self.w1(x)

        if self.params.w_lora:
            if 'up_proj' in self.params.target_modules:
                up_x = up_x + self.lora_w3(x)

            if 'gate_proj' in self.params.target_modules:
                gate_x = gate_x + self.lora_w1(x)

        down_input = F.silu(gate_x) * up_x
        out = self.w2(down_input)

        if self.params.w_lora:
            if 'down_proj' in self.params.target_modules:
                out = out + self.lora_w2(down_input)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            args=args
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # self.tok_embeddings = ParallelEmbedding(
        #     params.vocab_size, params.dim, init_method=lambda x: x
        # )
        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim,
        )

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)


        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        # self.output = ColumnParallelLinear(
        #     params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        # )
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False,)

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.grad_ckpt = params.grad_ckpt

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, labels: torch.Tensor):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        start_pos = 0
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device, dtype=torch.float32
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        if self.grad_ckpt == False or self.training==False:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            h = self.norm(h)
            output = self.output(h).float()

        else:
            h = self.layers[0](h, start_pos, freqs_cis, mask)            
            output = self._ckpt_forward(h, start_pos, freqs_cis, mask)
        
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        return c_loss
    
    def _ckpt_forward(self, h, start_pos, freqs_cis, mask):
        for layer in self.layers[1:]:
            h = torch.utils.checkpoint.checkpoint(layer, h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = torch.utils.checkpoint.checkpoint(self.output, h).float()

        return output

    @torch.inference_mode()
    def forward_only(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            # mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            # mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
            mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=h.device, dtype=torch.float32), 1)#.to(dtype=torch.bfloat16)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        
        output = self.output(h[:, -1, :])  # only compute last logits

        return output.float()


    def set_lora_trainable(self):
        param_lora  = []
        # adapter = []

        for name, param in self.named_parameters():
            # if any(n in name for n in adapter):
            #     param.requires_grad = True
            #     param.data = param.data.float()
            #     param_adapter.append(param)
            if "lora" in name:
                param.requires_grad = True
                # param.data = param.data.float()
                param_lora.append(param)
            else:
                param.requires_grad = False

        return param_lora

    def set_all_frozen(self):
        # adapter = []

        for name, param in self.named_parameters():
            # if any(n in name for n in adapter):
            #     param.requires_grad = True
            #     param.data = param.data.float()
            #     param_adapter.append(param)
            param.requires_grad = False


    def set_gate_trainable(self):
        param_gate  = []
        # adapter = []

        for name, param in self.named_parameters():
            # if any(n in name for n in adapter):
            #     param.requires_grad = True
            #     param.data = param.data.float()
            #     param_adapter.append(param)
            if "gate" in name:
                param.requires_grad = True
                # param.data = param.data.float()
                param_gate.append(param)
            else:
                param.requires_grad = False

        return param_gate
    
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print("trainable param. - {}\%".format((trainable_params/all_param)*100))
        
        return trainable_params, all_param
    
    def lora_state_dict(self):
        """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

        Args:
            model: model with LoRA layers
            bias: 
                ``"none"``: state dict will not store bias weights,
                ``"lora_only"``: state dict will store bias weights only from LoRA layers,
                ``"all"``: state dict will store all bias weights.

        Returns:
            Weights and biases of LoRA layers

        Raises:
            NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
        """
        my_state_dict = self.state_dict()
        
        return {k: my_state_dict[k] for k in my_state_dict if 'lora' in k}
    
    def gate_state_dict(self):
        """Return state_dict with weights of LoRA's A and B matrices and with biases depending on the `bias` value.

        Args:
            model: model with LoRA layers
            bias: 
                ``"none"``: state dict will not store bias weights,
                ``"lora_only"``: state dict will store bias weights only from LoRA layers,
                ``"all"``: state dict will store all bias weights.

        Returns:
            Weights and biases of LoRA layers

        Raises:
            NotImplementedError: if `bias` not in ["none", "lora_only", "all"]
        """
        my_state_dict = self.state_dict()
        
        return {k: my_state_dict[k] for k in my_state_dict if 'gate' in k}
    

    def reset_lora_parameters(self):
        for m in self.modules():
            if isinstance(m, LoraInjectedLinear):
                m.re_init_param()

    def reset_gate_parameters(self):
        for m in self.modules():
            if isinstance(m, LoraInjectedLinear) and m.w_gate:
                m.input_gate.data.zero_()
                # torch.nn.init.kaiming_uniform_(m.input_gate, nonlinearity='sigmoid')
                # torch.nn.init.xavier_normal_(m.input_gate)
                # torch.nn.init.xavier_uniform_(m.input_gate)

    def set_trainable_modules(self, target: Tuple[str]):

        # param_lora  = []
        # adapter = []

        self.params.target_modules = target
        
        namedict = {
            'q_proj': 'lora_wq',
            'k_proj': 'lora_wk',
            'v_proj': 'lora_wv',
            'o_proj': 'lora_wo',
            'up_proj': 'lora_w3',
            'down_proj': 'lora_w2',
            'gate_proj': 'lora_w1',
        }

        module_list = []
        for i in target:
            module_list.append(namedict[i])

        for name, param in self.named_parameters():
            # if any(n in name for n in adapter):
            #     param.requires_grad = True
            #     param.data = param.data.float()
            #     param_adapter.append(param)

            requires_grad = False
            for j in module_list:
                if j in name:
                    requires_grad = True

            param.requires_grad = requires_grad
            # param.data = param.data.float()
            # param_lora.append(param)

        # return param_lora


    def merge_lora_parameters(self, ratio=1.0):
        # Assuming `self` is the model containing both LLM and LoRA parameters.
        # `ratio` is a hyperparameter to scale the effect of LoRA modifications.

        # Create a mapping from LoRA parameter names to their corresponding original module.
        name_to_module = {}
        for name, param in self.named_parameters():
            if "lora" in name and "gate" not in name:
                # Extract the base module name by removing the LoRA specific suffix.
                # print(name)
                module_name = name.rsplit('.', 2)[0]  # This removes the last two components (e.g., "lora_down.weight")
                if module_name not in name_to_module:
                    # Directly map to the parameter object for now.
                    # This assumes the naming convention matches directly with how the parameters are stored in the model.
                    # print(module_name)
                    name_to_module[module_name] = self.get_parameter(name.rsplit('.', 2)[0].replace('lora_', '')+'.weight')
        # print(name_to_module)
        # Now, iterate over the LoRA parameters and merge them into the original model weights.
        for lora_name, lora_param in self.named_parameters():
            if "lora_down" in lora_name:
                up_name = lora_name.replace("lora_down", "lora_up")
                base_name = lora_name.rsplit('.', 2)[0]  # Get the base module name.

                # Get the corresponding LoRA "up" parameter and the original module parameter.
                up_param = self.get_parameter(up_name)
                base_param = name_to_module[base_name]

                # Calculate the merged weight.
                dim = lora_param.size(0)
                alpha = self.params.lora_alpha  # In actual use, alpha might be defined differently or be a learned parameter.
                scale = alpha / dim
                merged_weight = base_param.data + ratio * (up_param @ lora_param) * scale

                # Update the original module weight.
                base_param.data = merged_weight

        self.reset_lora_parameters()
        self.set_lora_trainable()

    def get_new_lora(self, lora_path_list: list, 
                     gate_path_list: list, 
                     input_ids: torch.Tensor, 
                     labels: torch.Tensor, 
                     batch_size: int, 
                     topk: int, 
                     epoch: int, 
                     temperature: float, 
                     sample: bool=False, 
                     sample_topk: int=None,
                     sample_temperature: float=None,
                     sample_top_p: float=None,
                     shared_ratio: float=1.0,
                     save_path: str=None):

        self.gate_layer_norm = nn.LayerNorm(
            self.params.dim, elementwise_affine=False
        )
        self.input_layer_norm = nn.LayerNorm(
            self.params.dim, elementwise_affine=False
        )
    
        lora_cache = {}
        for lora_id in lora_path_list:
            lora_cache[lora_id] = torch.load(lora_id, map_location='cpu')

        gate_cache = {}
        for gate_id in gate_path_list:
            gate_cache[gate_id] = torch.load(gate_id, map_location='cpu')

        name2module = {}
        lora_paths = list(lora_cache.keys())
        gate_paths = list(gate_cache.keys())

        for module_name, params in gate_cache[gate_paths[0]].items():
            name = module_name.replace('.input_gate', '')
            name2module[name] = {
                'gate': [],
                'loras': [],
            }

            for i in range(len(gate_cache)):

                lora_i = lora_cache[lora_paths[i]]
                gate_i = gate_cache[gate_paths[i]]

                gate_param = gate_i[module_name]
                lora_param = {}

                for n, m in lora_i.items():
                    if name in n:
                        lora_param[n] = m
                # print(name2module)
                # print(name)
                # print(name2module[name])
                
                name2module[name]['gate'].append(gate_param)
                name2module[name]['loras'].append(lora_param)
                
            name2module[name]['gate'] = torch.stack(name2module[name]['gate'], dim=0)
            
            if shared_ratio !=0 :
                anchor_cnt = name2module[name]['gate'].size(0)
                selected_idx = random.sample(range(anchor_cnt), int(anchor_cnt * shared_ratio))
                name2module[name]['gate'] = name2module[name]['gate'][selected_idx]
                new_loras = []

                for j in selected_idx:
                    new_loras.append(name2module[name]['loras'][j])
                
                name2module[name]['loras'] = new_loras

                # print(name2module[name]['gate'].size())

        for n, m in self.named_modules():
            if len(n.split('.'))==4 and 'lora' in n:
                m.name = n
                m.return_flag = True
                m.corpus = name2module[n]
                m.topk = topk
                m.temperature = temperature
                m.sample = sample
                m.sample_topk = sample_topk
                m.sample_temperature = sample_temperature
                m.sample_top_p = sample_top_p

        lora_param_list = []
        if save_path is not None:
            lora_record_dict = defaultdict(list)

        start_time = time.time()
        for _ in range(epoch):
            N = input_ids.size(0)
            indices = torch.randperm(N)
            shuffled_input_ids = input_ids[indices]
            shuffled_labels = labels[indices]

            input_batches = shuffled_input_ids.split(batch_size, dim=0)
            label_batches = shuffled_labels.split(batch_size, dim=0)

            for i in range(len(input_batches)):
                input = input_batches[i]
                label = label_batches[i]

                for n, m in self.named_modules():
                    if len(n.split('.'))==4 and 'lora' in n:
                        m.return_flag = True
                        m.label = label

                with torch.no_grad():
                    self.forward(input, label)
                
                for n, m in self.named_modules():
                    if len(n.split('.'))==4 and 'lora' in n:
                        if save_path is not None:
                            lora_record_dict[m.name].append(m.record)

                lora_param_list.append(self.lora_state_dict())
        
        # print(len(lora_param_list))
        
        keys = lora_param_list[0].keys()
        final_dict = {}
        weights = [(1/len(lora_param_list)) for _ in range(len(lora_param_list))]
        
        for i in range(len(lora_param_list)):
            lora_state_dict = lora_param_list[i]

            if i==0:
                for key in keys:
                    final_dict[key] = weights[i] * lora_state_dict[key].cuda()
            else:
                for key in keys:
                    final_dict[key] = final_dict[key] + weights[i] * lora_state_dict[key].cuda()

        for key, data in final_dict.items():
                param = self.get_parameter(key)
                param.data = data

        end_time = time.time()
        total_time = end_time - start_time

        if save_path is not None:
            # with open(os.path.join(save_path, ), 'w') as f:
            #     json.dump(lora_record_dict, f)

            torch.save(lora_record_dict, os.path.join(save_path, "ckpt.pt"))
        
        return total_time
        # def all_set():
        #     for n, m in self.named_modules():
        #         if len(n.split('.'))==4 and 'lora' in n and m.return_flag==True:
        #             return False
            
        #     return True
                    
        # while not all_set():
        #     input_batches = input_ids.split(batch_size, dim=0)
        #     label_batches = labels.split(batch_size, dim=0)

        #     weights_list = []

        #     for i in range(len(input_batches)):
        #         input = input_batches[i]
        #         label = label_batches[i]
                    
        #         for n, m in self.named_modules():
        #             if len(n.split('.'))==4 and 'lora' in n:
        #                 m.label = label

        #         try:
        #             with torch.no_grad():
        #                 self.forward(input, label)

        #         except EarlyReturnException as e:
        #             hidden_state = e.output['hidden_state'] # (batch_size, seq_len, dim)
        #             name = e.output['name']
        #             # print(name)
        #             target_module = name2module[name] # layers.30.attention.lora_wk

        #             hidden_state = self.input_layer_norm(hidden_state) # (batch_size, seq_len, dim)
        #             gate_state = self.gate_layer_norm(target_module['gate']).cuda() # (100, dim)

        #             scores = torch.matmul(hidden_state, gate_state.T)
        #             scores = scores * math.sqrt(1 / self.params.dim) # (batch_size, seq_len, 100)

        #             label_mask = label.gt(0)
        #             scores[~label_mask] = 0

        #             scores = torch.mean(scores, dim=-2) # (batch_size, 100)
        #             # weights = torch.softmax(scores, dim=-1) # (batch_size, 100)

        #             weights_list.append(scores) # (batch_size, 100)
            
        #     module = self.get_submodule(name)
        #     module.return_flag = False

        #     weights_all = torch.cat(weights_list, dim=0).mean(dim=0) # (100)
        #     weights_topk_value, weights_topk_idx = torch.topk(weights_all, k=topk)
        #     weights_sfm = torch.softmax(weights_topk_value, dim=0)
        #     print(weights_sfm)
        #     # print(weights_topk_idx)
        #     all_lora = target_module['loras']
        #     keys = all_lora[0].keys()
        #     final_state = {}

        #     for j in range(topk):
        #         lora = all_lora[weights_topk_idx[j]]
        #         if j == 0:
        #             for key in keys:
        #                 final_state[key] = weights_sfm[j] * lora[key].cuda()
        #         else:
        #             for key in keys:
        #                 final_state[key] = final_state[key] + weights_sfm[j] * lora[key].cuda()

        #     for key, data in final_state.items():
        #         param = self.get_parameter(key)
        #         param.data = data
            
            
                
                





        