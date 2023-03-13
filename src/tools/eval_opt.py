###

import torch
from transformers import AutoTokenizer, OPTForCausalLM, PreTrainedTokenizer

from classes.discord_dataset import DiscordDataset
from config import paths

model_file = paths.MODEL_DIR / "opt2.7" / "test_1.bin"

###

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)  # type: ignore
model = OPTForCausalLM.from_pretrained(
    "facebook/opt-2.7b",
    # load_in_8bit=True,
    # torch_dtype=torch.float16,
    max_memory={0: "19GiB", "cpu": "99GiB"},
    device_map="auto",
)  # type: ignore
state_dict = torch.load(model_file)
print("\n".join(list(state_dict.keys())[:10]))
state_dict["lm_head.weight"] = state_dict["model.decoder.embed_tokens.weight"]
model.load_state_dict(state_dict)  # type: ignore

###

# ds = DiscordDataset.load(tokenizer, sequence_length)
cases = [
    "Literal genie: The LLaMA language model is",
]

for c in cases:
    inputs = tokenizer(c, return_tensors="pt").input_ids.to(model.device)  # type: ignore
    generate_ids = model.generate(inputs, min_length=100, max_length=300, repetition_penalty=1.3)  # type: ignore
    outputs = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print((outputs.replace(c, c + "\n***\n")))
    print("---\n\n")
