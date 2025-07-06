import torch
import tiktoken
from lecture_25 import GPTModel, generate_text_simple
from lecture_26 import GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text

model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("gpt124m.pth", map_location="cpu"))
model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M
)

print("Output (default):\n", token_ids_to_text(token_ids, tokenizer))

