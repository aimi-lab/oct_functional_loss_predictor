
from enum import Enum


class SampleContent(Enum):
    INPUT_IMG = 'input_img'
    INPUT_TENSOR = 'input_tensor'
    TARGET = 'target'
    META = 'meta'


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __str__(self):
        return self.value

    def __repr__(self):
        return str(self)

    @classmethod
    def from_str(cls, split: str):
        return cls(split)

    @classmethod
    def from_num(cls, num: int):
        return cls(num)
    
class Laterality(Enum):
    LEFT = 0
    RIGHT = 1

    @staticmethod
    def from_string(s: str):
        left_strings = ["left", "l", "0", "lft", "os"]
        right_strings = ["right", "r", "1", "rght", "od"]

        if s.lower() in left_strings:
            return Laterality.LEFT
        elif s.lower() in right_strings:
            return Laterality.RIGHT
        else:
            raise ValueError(f"Unknown laterality: {s}")
    
    @staticmethod
    def from_int(i: int):
        if i == 0:
            return Laterality.LEFT
        elif i == 1:
            return Laterality.RIGHT
        else:
            raise ValueError(f"Unknown laterality: {i}")