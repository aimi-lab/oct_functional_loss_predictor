
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
    L = "left"
    R = "right"
    U = "unknown"

    @classmethod
    def from_str(cls, text: str):
        text = text.lower().strip()
        if text in ["left", "l", "os"]:
            return cls("left")
        elif text in ["right", "r", "od"]:
            return cls("right")
        else:
            return cls("unknown")

    @classmethod
    def from_int(cls, num: int):
        if num == 0:
            return cls("left")
        elif num == 1:
            return cls("right")
        else:
            return cls("unknown")
