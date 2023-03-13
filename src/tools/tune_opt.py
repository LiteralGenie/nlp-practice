from datetime import datetime

import deepspeed
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM, PreTrainedTokenizer

from classes.discord_dataset import DiscordDataset
from config import paths

###


model_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
out_dir = paths.CHECKPOINT_DIR / "opt2.7"
sequence_length = 1024
batch_size = 1
test_split = 0.1
epochs = 10

###

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)  # type: ignore
model: OPTForCausalLM = OPTForCausalLM.from_pretrained("facebook/opt-2.7b")  # type: ignore

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# generate_ids = model.generate(inputs.input_ids, max_length=30)  # type: ignore
# outputs = tokenizer.batch_decode(
#     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )[0]

# print(outputs)


ds = DiscordDataset.load(tokenizer, sequence_length)
test_size = round(len(ds) * test_split)
train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

config = dict(
    train_micro_batch_size_per_gpu=1,
    gradient_accumulation_steps=1,
    fp16=dict(
        enabled=True,
    ),
    optimizer=dict(
        type="Adam",
        params=dict(
            lr=2e-5,
            # betas="auto",
            # eps="auto",
            # weight_decay="auto",
        ),
    ),
    zero_optimization=dict(
        stage=2,
        offload_optimizer=dict(
            device="cpu",
            pin_memory=True,
        ),
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size=2e8,
        overlap_comm=True,
        contiguous_gradients=True,
    ),
    steps_per_print=1e10,
)
params = None
model_engine, optimizer, _, _ = deepspeed.initialize(
    config=config,
    model=model,
    model_parameters=params,
)

###


def get_model_name(id, ds, acc, loss, epoch):
    return f"opt_{ds.name.lower()}_{id}_{epoch:02}_{acc*100:.2f}_{loss:.4f}"


loss_fn = torch.nn.CrossEntropyLoss()

for epoch_idx in range(epochs):
    print("epoch", epoch_idx)

    pbar = tqdm(train_dl)
    losses = []
    for step, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(model_engine.local_rank)
        labels = inputs

        outputs = model_engine(input_ids=inputs, labels=labels)
        model_engine.backward(outputs.loss)
        model_engine.step()

        losses.append(outputs.loss.item())
        if len(losses) > 20:
            losses.pop(0)
        avg_loss = sum(losses) / len(losses)

        pbar.set_description(f"{epoch_idx:02} | {step} | {avg_loss:.2f}")

        if step and step % 300 == 0:
            # Save
            name = get_model_name(model_id, ds, 0, avg_loss, step)
            print("saving", name)
            model_engine.save_checkpoint(out_dir, name)
