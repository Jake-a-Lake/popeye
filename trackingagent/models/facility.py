from enum import Enum


class GetAttr(type):
    def __getitem__(cls, x):
        return getattr(cls, x)


class Facility(Enum):

    # COMEmployee = 1
    # MAC = 2
    # MCPN = 3
    # PepperGarageNW = 4
    # PepperGarageSouth = 5
    # PepperN = 6

    # NetSenseLotImportJob,
    Snapshot = 1
    


class FacilityStatus(Enum):
    Empty = 1
    Empty_Filling = 2
    Available_Emptying = 3
    Available = 4
    Aailable_Filling = 5
    Full_Emptying = 6
    Full = 7
