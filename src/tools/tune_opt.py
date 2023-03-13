from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import deepspeed
import fire
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM, PreTrainedTokenizer

from classes.discord_dataset import DiscordDataset
from config import paths

###


@dataclass
class _Params:
    batch_size: int = 1
    epochs: int = 10
    loss_window: int = 50
    lr: float = 2e-5
    model_id: str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_dir: str = str(paths.CHECKPOINT_DIR / "opt2.7")
    save_steps: int = 300
    sequence_length: int = 1024
    test_split: float = 0.1

    out_dir_fp: Path = field(init=False)

    def __post_init__(self):
        self.out_dir_fp = Path(self.out_dir)


###


def main(**kwargs: _Params):
    params = _Params(**kwargs)
    train(params)


def train(p: _Params):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("/media/anne/the_one/llama/models/conv/llama-7b/", use_fast=False)  # type: ignore
    model: OPTForCausalLM = OPTForCausalLM.from_pretrained("/media/anne/the_one/llama/models/conv/llama-7b/tokenizer.model")  # type: ignore

    ds = DiscordDataset.load(tokenizer, p.sequence_length)
    test_size = round(len(ds) * p.test_split)
    train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])

    train_dl = DataLoader(
        train_ds, batch_size=p.batch_size, shuffle=True, drop_last=True
    )
    test_dl = DataLoader(test_ds, batch_size=p.batch_size, shuffle=True, drop_last=True)

    config = dict(
        train_micro_batch_size_per_gpu=p.batch_size,
        gradient_accumulation_steps=1,
        fp16=dict(
            enabled=True,
        ),
        optimizer=dict(
            type="Adam",
            params=dict(
                lr=p.lr,
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
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=config,
        model=model,
    )

    ###

    def get_model_name(id, ds, acc, loss, epoch):
        return f"opt_{ds.name.lower()}_{id}_{epoch:02}_{acc*100:.2f}_{loss:.4f}"

    for epoch_idx in range(p.epochs):
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
            if len(losses) > p.loss_window:
                losses.pop(0)
            avg_loss = sum(losses) / len(losses)

            pbar.set_description(f"{epoch_idx:02} | {step} | {avg_loss:.2f}")

            if step and step % p.save_steps == 0:
                # Save
                name = get_model_name(p.model_id, ds, 0, avg_loss, step)
                print("saving", name)
                model_engine.save_checkpoint(p.out_dir, name)


if __name__ == "__main__":
    fire.Fire(main)
