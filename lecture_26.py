GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

import torch
import torch.nn as nn
from lecture_25 import GPTModel, generate_text_simple
import tiktoken

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

inputs = torch.tensor([
    [16833,  3626,  6100],  # ["every effort moves",
    [    40,  1107,  5881]   #  ["I really like"]
])

targets = torch.tensor([
    [ 3626,  6100,   345],  # [" effort moves you",
    [  107,   588, 11311]   #  [" really like chocolate"]
])

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)












