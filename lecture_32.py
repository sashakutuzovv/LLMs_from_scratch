import torch
import torch.nn as nn
import tiktoken
from lecture_25 import GPTModel #, generate_text_simple
from lecture_26 import GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text
from lecture_30 import generate
#import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from gpt_download import download_and_load_gpt2
import numpy as np

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
model_configs = {
    "gpt2-small (124M)":  {"emb_dim":  768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)":    {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pick the model you want to use
model_name = "gpt2-small (124M)"  # example
tokenizer = tiktoken.get_encoding("gpt2")

# Start from the base GPT_CONFIG_124M and update with your choice
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})
gpt = GPTModel(NEW_CONFIG)
gpt.eval()

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    # Embeddings
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params['blocks'])):
        # QKV weights
        q_w, k_w, v_w = np.split(
            params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # QKV biases
        q_b, k_b, v_b = np.split(
            params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # Attention output projection
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params['blocks'][b]['attn']['c_proj']['w'].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params['blocks'][b]['attn']['c_proj']['b'])

        # Feed-forward layers
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params['blocks'][b]['mlp']['c_fc']['w'].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params['blocks'][b]['mlp']['c_fc']['b'])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params['blocks'][b]['mlp']['c_proj']['w'].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params['blocks'][b]['mlp']['c_proj']['b'])

        # Layer norms
        gpt.trf_blocks[b].norm1.weight = assign(
            gpt.trf_blocks[b].norm1.weight,
            params['blocks'][b]['ln_1']['g'])
        gpt.trf_blocks[b].norm1.bias = assign(
            gpt.trf_blocks[b].norm1.bias,
            params['blocks'][b]['ln_1']['b'])
        gpt.trf_blocks[b].norm2.weight = assign(
            gpt.trf_blocks[b].norm2.weight,
            params['blocks'][b]['ln_2']['g'])
        gpt.trf_blocks[b].norm2.bias = assign(
            gpt.trf_blocks[b].norm2.bias,
            params['blocks'][b]['ln_2']['b'])

    # Final layer norm
    gpt.final_norm.weight = assign(gpt.final_norm.weight, params['g'])
    gpt.final_norm.bias = assign(gpt.final_norm.bias, params['b'])

    # Output head
    gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])

load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


