import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from classes.enwiki_dataset import EnwikiDataset

Lines = list[str]
VocabTally = dict[str, int]


class Nnlm(nn.Module):
    params_feature = 1000
    params_hidden = 500

    def __init__(self, vocab_size: int, lookback_count: int):
        super().__init__()

        self.vocab_size = vocab_size

        # number of previous words to peek at +1
        self.n = lookback_count + 1

        self.C = nn.Embedding(self.vocab_size, self.params_feature)

        # hidden layer
        self.H = nn.Linear((self.n - 1) * self.params_feature, self.params_hidden)
        self.H_act = nn.Tanh()

        # output layer weights
        self.U = nn.Linear(self.params_hidden, vocab_size)
        self.U_act = nn.Softmax(0)

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
    freq_thresh = 100
    batch_size = 64
    test_split = 0.1
    learning_rate = 0.001

    ds = EnwikiDataset.load(sequence_length, freq_thresh)
    test_size = round(len(ds) * test_split)
    train_ds, test_ds = random_split(ds, [len(ds) - test_size, test_size])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Nnlm(len(ds.vocab), sequence_length - 1)

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for i, (input, label) in enumerate(train_dl):
        input = input.to(device)
        label = label.to(device)

        pred = model(input)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            loss, current = loss.item(), (i + 1) * len(input)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(ds):>5d}]")
