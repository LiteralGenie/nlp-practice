from torch import nn
from torch.utils.data import DataLoader2
from torchtext import datasets

from config import paths


class Nnlm(nn.Module):
    def __init__(self):
        self.embeddings = nn.Embedding()  # C in paper


if __name__ == "__main__":
    print("Loading dataset")
    pipe = datasets.EnWik9(str(paths.DATA_DIR))

    loader = DataLoader2(pipe, shuffle=True, drop_last=True)

    for i, x in enumerate(iter(loader)):
        if i == 100:
            break

        print(i, x)
