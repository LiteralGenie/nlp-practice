from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import deepspeed
import fire
import torch
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, OPTForCausalLM, PreTrainedTokenizer

from classes.discord_dataset import DiscordDataset
from config import paths

###


@dataclass
class _Params:
    batch_size: int = 1
    batch_size_test: int | None = None
    epochs: int = 10
    loss_window: int = 50
    lr: float = 1e-5
    model_id: str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    out_dir: str = str(paths.CHECKPOINT_DIR / "opt2.7")
    resume_from: str | None = str(paths.CHECKPOINT_DIR / "opt2.7")
    save_steps: int = 200
    sequence_length: int = 1024
    test_split: float = 0.00005
    use_cached_data: bool = True

    out_dir_fp: Path = field(init=False)
    resume_from_fp: Path | None = field(init=False)

    def __post_init__(self):
        self.out_dir_fp = Path(self.out_dir)
        self.resume_from_fp = Path(self.resume_from) if self.resume_from else None


###


def main(**kwargs: _Params):
    params = _Params(**kwargs)
    train(params)


def train(p: _Params):
    logger.info("Loading models")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False)  # type: ignore
    model: OPTForCausalLM = OPTForCausalLM.from_pretrained("facebook/opt-2.7b")  # type: ignore

    logger.info("Loading dataset")
    ds = DiscordDataset.load(tokenizer, p.sequence_length, from_cache=p.use_cached_data)

    test_size = round(len(ds) * p.test_split)
    train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])
    train_dl = DataLoader(
        train_ds,
        batch_size=p.batch_size_test or p.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_dl = DataLoader(test_ds, batch_size=p.batch_size, shuffle=True, drop_last=True)
    logger.info(f"{len(train_ds):,} training samples, {len(test_ds):,} test samples")

    config = dict(
        train_micro_batch_size_per_gpu=p.batch_size,
        gradient_accumulation_steps=1,
        fp16=dict(
            enabled=True,
        ),
        optimizer=dict(
            type="AdamW",
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
    logger.info("Initializing engine")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        config=config,
        model=model,
    )

    if p.resume_from_fp:
        logger.info(f"Resuming from {p.resume_from_fp}")
        model_engine.load_checkpoint(load_dir=p.resume_from_fp)

    ###

    def get_model_name(id, ds, acc, loss, epoch):
        return f"opt_{ds.name.lower()}_{id}_{epoch:02}_{acc:.2f}_{loss:.4f}"

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
                # Eval
                with torch.no_grad():
                    val_loss = []
                    val_correct = 0
                    val_count = 0
                    val_loss_avg = 0
                    val_acc_avg = 0

                    pbar_val = tqdm(test_dl, leave=True)
                    for i, (inputs, labels) in enumerate(pbar_val):
                        inputs = inputs.to(model_engine.local_rank)
                        labels = inputs

                        outputs = model_engine(input_ids=inputs, labels=labels)

                        val_count += len(outputs.logits)
                        val_loss.append(outputs.loss.item())
                        val_correct += torch.sum(
                            outputs.logits[:, -1].argmax(-1) == labels[:, -1]
                        ).item()

                        val_loss_avg = sum(val_loss) / len(val_loss)
                        val_acc_avg = val_correct / val_count
                        pbar_val.set_description(
                            f"{epoch_idx:02} | {step} | {val_loss_avg=:.2f} | {val_acc_avg=:.2f}"
                        )

                logger.info(
                    f"{epoch_idx:02} | {step} | {val_loss_avg=:.2f} | {val_acc_avg=:.2f} | train_loss_avg={sum(losses) / len(losses):.2f}"
                )

                # Save
                name = get_model_name(p.model_id, ds, val_acc_avg, val_loss_avg, step)
                logger.info("saving " + name)
                model_engine.save_checkpoint(p.out_dir, name)


if __name__ == "__main__":
    fire.Fire(main)
