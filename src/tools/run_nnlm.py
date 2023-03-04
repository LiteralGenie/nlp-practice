from datetime import datetime
from random import random

import torch
from classes.diplomacy_dataset import DiplomacyDataset
from classes.enwiki_dataset import EnwikiDataset
from classes.simple_dataset import SimpleDataset
from classes.urban_dictionary_dataset import UrbanDictionaryDataset
from config import paths
from torch import LongTensor, nn
from torch.utils.data import DataLoader, random_split

Lines = list[str]
VocabTally = dict[str, int]


class Nnlm(nn.Module):
    params_feature = 2000
    params_hidden = 1000

    def __init__(self, vocab_size: int, lookback_count: int):
        super().__init__()

        self.vocab_size = vocab_size

        # number of previous words to peek at +1
        self.n = lookback_count + 1

        self.C = nn.Embedding(self.vocab_size, self.params_feature)

        # hidden layer
        self.H = nn.Linear((self.n - 1) * self.params_feature, self.params_hidden)
        self.H_act = nn.Tanh()

        # output layer -- should learn the index [0, vocab_size) of the next word
        self.U = nn.Linear(self.params_hidden, self.vocab_size)
        self.U_act = nn.Softmax(1)

    def forward(self, xs):
        # convert words to embeddings
        out = self.C(xs)

        # concat embeddings
        out = out.view(-1, (self.n - 1) * self.params_feature)

        # feed to hidden layer
        out = self.H(out)
        out = self.H_act(out)

        # feed to output layer
        out = self.U(out)
        out = self.U_act(out)

        return out


if __name__ == "__main__":
    sequence_length = 5
    freq_thresh = 5
    batch_size = 100
    test_split = 0.1
    learning_rate = 0.01
    momentum = 0.1
    epochs = 10000
    out_dir = paths.MODEL_DIR / "nnlm"
    model_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    model_name = (
        lambda id, ds, acc, loss, epoch: f"nnlm_{ds.name.lower()}_{id}_{epoch:02}_{acc*100:.2f}_{loss:.4f}.ckpt"
    )

    ds = DiplomacyDataset.load(sequence_length, freq_thresh)
    # ds = SimpleDataset(sequence_length, vocab_size=10)

    test_size = round(len(ds) * test_split)
    train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Nnlm(len(ds.vocab), sequence_length - 1)

    # Prep
    out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        val_acc = 0

        for i, (input, label) in enumerate(train_dl):
            input = input.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            pred = model(input)
            loss = loss_fn(pred, label)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if i % 100 == 0:
                print(
                    f"\r{epoch:03} | loss: {train_loss / ((i+1)):>7f} | {(i+1):,} / {len(train_dl):,}",
                    end="",
                )

        with torch.no_grad():
            for i, (input, label) in enumerate(test_dl):
                input = input.to(device)
                label = label.to(device)

                pred = model(input)
                loss = loss_fn(pred, label)
                val_loss += loss.item()
                n_correct = torch.sum(pred.argmax(1) == label)
                val_acc += n_correct / len(label)

            print(
                f"\r{epoch:03} | val acc: {val_acc * 100 / len(test_dl):>5.2f}% | val loss: {val_loss / len(test_dl):>7f} {'':>100}"
            )

            input, label = next(iter(test_dl))
            input = input.to(device)
            label = label.to(device)

            input_words = " ".join([ds.index_to_vocab[int(x)] for x in input[0]])
            label_word = ds.index_to_vocab[int(label[0])]
            print(f"\tval input: {input_words}")
            print(f"\tval label: {label_word}")

            pred = model(input)
            pred = pred.argmax(1)[0]
            pred_word = ds.index_to_vocab[int(pred)]
            print(f"\tval pred:  {pred_word}")

        if epoch % 10 == 0:
            out_file = out_dir / model_name(
                model_id, ds, val_acc / len(test_dl), val_loss / len(test_dl), epoch
            )
            torch.save(model.state_dict(), out_file)

    # Predict
    # model.eval()
    # model.load_state_dict(
    #     torch.load(
    #         "/media/anne/bottle/projs/python/nlp-practice/src/data/models/nnlm/nnlm_diplomacy_2023-03-03T22:00:50_10_7.8088.ckpt"
    #     )
    # )

    # with torch.no_grad():
    #     for i, (sample, label) in enumerate(test_dl):
    #         if i == 10:
    #             break

    #         sample = sample.to(device)
    #         label = label.to(device)

    #         sample_words = " ".join([ds.index_to_vocab[int(x)] for x in sample[0]])
    #         label_word = ds.index_to_vocab[int(label[0])]
    #         print(f"testing: {sample_words}")
    #         print(f"\t{label_word}")

    #         pred = model(sample)
    #         pred = pred.argmax(1)[0]
    #         pred_word = ds.index_to_vocab[int(pred)]
    #         print(f"\t{pred_word}")

    #     print("\n\n\n\n")
    #     pred_count = 10
    #     cases = [
    #         "i hate the rain it makes me sick but whatever i will make do if you could just not",
    #         "he is able to change his plans before <UNK> i'm assuming germany <UNK> out to you about attacking me",
    #         "<UNK> my dude this is getting me back into the game not gonna lie <UNK> start but im <UNK>",
    #     ]
    #     for c in cases:
    #         sample = []
    #         for w in c.split()[: sequence_length - 1]:
    #             idx = ds.vocab_to_index.get(w.lower(), ds.vocab_to_index["<UNK>"])
    #             sample.append(idx)

    #         print(c)
    #         print("\t", sample)

    #         preds = []
    #         for i in range(pred_count):
    #             sample_tensor = LongTensor([sample]).to(device)
    #             pred = model(sample_tensor)
    #             pred = pred.argmax(1)[0]
    #             pred_word = ds.index_to_vocab[int(pred)]
    #             preds.append(pred_word)

    #             sample = sample[1:] + [int(pred)]

    #         print("\t", " ".join(preds))
