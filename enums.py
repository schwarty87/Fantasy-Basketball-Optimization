"""
all enums used in the project
to get a dict of all enum values in a certain Enum, do:
all_values_in_enum = ClassThatInheritsFromEnum._membermap_
"""

from enum import Enum, auto
from machine_learning.registries import enums_registry


class DataSplitType(Enum):
    train = auto()
    validation = auto()
    test = auto()

    @classmethod
    def all(cls):
        return list(cls)


class StatsTypes(Enum):
    player = auto()
    team = auto()
    all = [player, team]


class BoxscoreTypes(Enum):
    Base = auto()
    Advanced = auto()
    Usage = auto()
    FourFactors = auto()


class Browsers(Enum):
    CHROME = auto()
    FIREFOX = auto()


@enums_registry.register
class Sites(Enum):
    YAHOO = auto()
    DRAFTKINGS = auto()
    FANDUEL = auto()

    @classmethod
    def list_members(cls):
        return [m for m in cls]


if __name__ == '__main__':
    dst = DataSplitType._member_map_
    print(dst)
