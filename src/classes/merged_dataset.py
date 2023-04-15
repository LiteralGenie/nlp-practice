from torch.utils.data import Dataset


class MergedDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.name = "_".join(getattr(ds, "name", "??") for ds in self.datasets)

        idx = 0
        self.start_idxs = [0]
        for ds in self.datasets:
            idx += len(ds)
            self.start_idxs.append(idx)

    def __getitem__(self, index):
        for ds_idx in range(len(self.start_idxs[:-1])):
            start = self.start_idxs[ds_idx]
            end = self.start_idxs[ds_idx + 1]

            if index >= start and index < end:
                sample_idx = index - start
                sample = self.datasets[ds_idx][sample_idx]
                return sample

    def __len__(self):
        return self.start_idxs[-1]
