from torch import nn

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

        # hidden layer weights
        self.H = nn.Linear((self.n - 1) * self.params_feature, self.params_hidden)

        # output layer weights
        self.U = nn.Linear(self.params_hidden, self.params_feature)

    def forward(self, xb):
        # convert words to embeddings
        out = self.C(xb)

        # concat embeddings
        out = out.view(-1, (self.n - 1) * self.params_feature)

        # feed to hidden layer
        out = self.H(out)
        out = nn.Tanh(out)

        # feed to output layer
        out = self.U(out)
        out = nn.Softmax(out)

        return out


if __name__ == "__main__":
    sequence_length = 5
    freq_thresh = 100

    ds = EnwikiDataset.load(sequence_length, freq_thresh)
    model = Nnlm(len(ds.vocab), sequence_length)
