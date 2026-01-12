import math
import random
import logging
from collections import OrderedDict
from itertools import chain
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)

@dataclass
class Split:
    """
    A class to represent a split for balancing the number of entries in each split.

    Args:
    name (str): The name of the split (e.g. 'train', 'val', 'test')
    target_size (int): The desired number of entries in the split
    entries (OrderedDict): A dictionary with the labels as key and the indices of the entries with the label in the array as values.

    Attributes:
    name (str): The name of the split (e.g. 'train', 'val', 'test')
    target_size (int): The desired number of entries in the split
    entries (OrderedDict): A dictionary with the labels as key and the indices of the entries with the label in the array as values.
    """

    name: str
    target_size: int
    entries: OrderedDict

    @property
    def indices(self):
        return list(chain.from_iterable(self.entries.values()))

    @property
    def n_indices(self):
        return len(self.indices)

    @property
    def size_mismatch(self):
        return self.n_indices - self.target_size

    def sort_by_n_indices(self):
        self.entries = OrderedDict(
            sorted(self.entries.items(), key=lambda e: len(e[1]))
        )

def train_val_test_split(
    array,
    val_size: float,
    test_size: float,
    random_state: int = None,
    shuffle: bool = True,
    separate=None,
):
    """
    Splits the input array into three subsets: train, validation, and test set.
    The size of the validation and test set is determined by the val_size and test_size parameters.
    
    Args:
        array (list): The input array to be split.
        val_size (float): The size of the validation set as a fraction of the input array.
        test_size (float): The size of the test set as a fraction of the input array.
        random_state (int, optional): The random seed to ensure reproducibility.
        shuffle (bool, optional): Whether to shuffle the array before splitting.

    Returns:
        tuple: A tuple containing three lists: train, validation, and test split.
    """

    if separate is None:
         return _regular_split(array, val_size, test_size, random_state, shuffle)
    else:
        return _separated_split(
            array, separate, val_size, test_size, random_state, shuffle
        )
       

def _regular_split(
    array,
    val_size: float,
    test_size: float,
    random_state: int = None,
    shuffle: bool = True,
):
    
    if shuffle and random_state:
        random.Random(random_state).shuffle(array)
    elif shuffle:
        random.shuffle(array)

    train_size, val_size, test_size = _split_size(len(array), val_size, test_size)

    val = array[:val_size]
    test = array[val_size : val_size + test_size]
    train = array[val_size + test_size :]

    return train, val, test


def _separated_split(
    array: list,
    separate: list,
    val_size: float,
    test_size: float,
    random_state: int = None,
    shuffle: bool = True,
    max_iter: int = 10,
):
    """
    Splits the array into three subsets ensures that entries with the same
    label are only present in a single split (e.g. to ensure separation of patients.)

    Input is first split into subsets based on the labels in the separate list.
    Subsequently, the label splits are expanded by adding entries with the same label to the designated split.

    Entries with the same label are moved between the splits to reach the desired the number of entries in each split.

    """
    SPLIT_NAMES = "train", "val", "test"

    assert len(array) == len(separate), "Array and separate must have the same length"

    target_sizes = _split_size(len(array), val_size, test_size)

    # get the indices of entries with the same label
    lbl_2_indices = _get_indices_dict(separate)
    separation_labels = list(lbl_2_indices.keys())

    # split into subsets based on the label -> each label will only be in one split
    lbl_splits = _regular_split(
        separation_labels,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    splits = list()
    for i, name in enumerate(SPLIT_NAMES):
        splits.append(
            Split(name, target_sizes[i], _get_sub_dict(lbl_2_indices, lbl_splits[i]))
        )

    splits = _balance_splits(splits, max_iter)

    # Balancing moves the splits, bring them back into train, val, test order
    splits = sorted(splits, key=lambda s: SPLIT_NAMES.index(s.name))

    for i, name in enumerate(SPLIT_NAMES):
        LOGGER.debug(f"{name}: {splits[i].n_indices}")
        splits[i] = [array[idx] for idx in splits[i].indices]

        # balancing and expanding patly sorts the splits, shuffle them again before returning
        if shuffle and random_state:
            random.Random(random_state).shuffle(splits[i])
        elif shuffle:
            random.shuffle(splits[i])

    return splits


def _get_indices_dict(array: list) -> dict:
    """
    Returns a dictionary with the unique entries of the array as keys and the indices of these entries as values.
    """
    indices_dict = {label: list() for label in array}
    for idx, label in enumerate(array):
        indices_dict[label].append(idx)
    return indices_dict


def _get_sub_dict(d: dict, selection: list) -> dict:
    """
    Returns a dictionary with the keys from the selection list and the corresponding values from the input dictionary.
    """
    return {k: v for k, v in d.items() if k in selection}


def _split_size(ds_size: int, val_size: float, test_size: float):
    val_size = math.floor(ds_size * val_size)
    test_size = math.floor(ds_size * test_size)
    train_size = ds_size - val_size - test_size
    return train_size, val_size, test_size


def _balance_splits(splits: list[Split], max_iterations: int = 10) -> list[Split]:
    """
    Balances the splits by moving entries between the splits.
    The entries are moved from the split with the most entries to the split with the least entries.
    All labels of a label are move at once to enure separation of labels.
    """

    for split in splits:
        split.sort_by_n_indices()

    for i in range(max_iterations):
        splits = sorted(splits, key=lambda s: s.size_mismatch)
        entries_deficit = splits[0]
        entries_surplus = splits[-1]

        key, val = entries_surplus.entries.popitem(last=False)

        entries_deficit.entries[key] = val
        entries_deficit.sort_by_n_indices()

        max_mismatch = max([s.size_mismatch for s in splits])

        if max_mismatch == 0:
            LOGGER.debug("Balancing complete")
            break

        if i == max_iterations - 1:
            LOGGER.warning("Balancing did not converge")

    return splits


if __name__ == "__main__":

    def test_1():
        """
        Test that the splits do not have common labels, when separate is set.
        """
        single_labels = list(range(20))
        double_labels = 2 * list(range(20, 25))
        tripple_labels = 3 * list(range(25, 30))
        labels = single_labels + double_labels + tripple_labels

        train, val, test = train_val_test_split(labels, 0.15, 0.15, random_state=42, shuffle=True, separate=labels)

        assert set(train) & set(val) == set(), "Train and val split have common labels"
        assert set(train) & set(test) == set(), "Train and test split have common labels"
        assert set(val) & set(test) == set(), "Val and test split have common labels"
        print("Test 1 passed")

        LOGGER.debug(f'Labels (input): {labels}')
        LOGGER.debug(f'Train (output): {train}')
        LOGGER.debug(f'Val (output): {val}')
        LOGGER.debug(f'Test (output): {test}')

    test_1()
