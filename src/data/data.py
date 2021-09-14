import numpy as np
from torch.utils.data import Dataset


class LibTXTData(Dataset):
    def __init__(self, root, config):
        data = np.loadtxt(root)

        if config.out_d == 1:
            self.feat, self.label = (
                data[:, 0 : config.feat_d],
                data[:, config.feat_d],
            )
        else:
            self.feat, self.label = (
                data[:, 0 : config.feat_d],
                data[:, config.feat_d : config.feat_d + config.out_d],
            )
        del data
        self.feat = self.feat.astype(np.float32)
        self.label = self.label.astype(np.float32)

    def __getitem__(self, index):
        return self.feat[index, :], self.label[index]

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        for data in zip(self.feat, self.label):
            yield data