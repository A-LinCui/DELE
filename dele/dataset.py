from typing import List

from torch.utils.data import Dataset


class ArchDataset(Dataset):
    """
    Architecture-performance dataset for vanilla predictor training.

    Args:
        data (list): A list of architecture-performance data.
                     Each item in the list should be of the following form: [arch, perf],
                     where `arch` is the architecture representation and `perf` is the 
                     actual performance (float).
    """

    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        arch, performance = self.data[idx]
        return arch, performance


class LFArchDataset(Dataset):
    """
    Architecture-performance dataset with a type of low-fidelity information for predictor training.

    Please be sure to normalize each low-fidelity information to [0., 1.] in advance, 
    if you train the predictor with the regression loss (compare = False).

    Args:
        data (list): A list of architecture-performance data.
                     Each item in the list should be of the following form: [arch, perf, lf],
                     where `arch` is the architecture representation, `perf` is the actual
                     performance (float) and `lf` is the low-fidelity estimations 
                     (Dict[str, float]). 
        low_fidelity_type (str): The utilized low-fidelity information type.
    """

    def __init__(self, data: list, low_fidelity_type: str):
        self.data = data
        self.low_fidelity_type = low_fidelity_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        arch, performance, all_low_fidelity_estimations = self.data[idx]
        return arch, performance, all_low_fidelity_estimations[self.low_fidelity_type]


class MultiLFArchDataset(Dataset):
    """
    Architecture-performance dataset for dynamic ensemble predictor training.

    Args:
        data (list): A list of architecture-performance data.
                     Each item in the list should be of the following form: [arch, perf, lf],
                     where `arch` is the architecture representation, `perf` is the actual
                     performance (float) and `lf` is the low-fidelity estimations 
                     (Dict[str, float]).
        low_fidelity_types (List[str]): Utilized low-fidelity information types.
    """

    def __init__(self, data: list, low_fidelity_types: List[str]):
        self.data = data
        self.low_fidelity_types = low_fidelity_types

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        arch, performance, all_low_fidelity_estimations = self.data[idx]
        low_fidelity_performances = {
            _type: all_low_fidelity_estimations[_type]
            for _type in self.low_fidelity_types
        }
        return arch, performance, low_fidelity_performances
