import pyodbc
from datetime import datetime, timedelta
from repositories.facilitycountrepo import FacilityCountRepository as repository


class FacilityCountSaveService:
    def __init__(self, facilityCountObject):
        self.data = facilityCountObject

    # this type of logic moving to view
    def deriveLotStatus(facilityId, carCount):
        # self.car_count = lot_count
        facilityId = facilityId  # self.data["FACILITY_ID"]
        carCount = carCount  # self.data["CARCOUNT"]
        lot_status = 0

        if facilityId == 1:  # COMEmployee hypothetical lot capacity of 150
            capacity = 150
            occupy_pc = (carCount / capacity) * 100
            if occupy_pc > 90 and occupy_pc <= 100:
                lot_status = 7  # full
            elif occupy_pc > 70 and carCount <= 89:
                lot_status = 5  # avail fulling
            elif occupy_pc > 60 and occupy_pc <= 69:
                lot_status = 4
            elif occupy_pc > 50 and occupy_pc <= 59:
                lot_status = 3  # avail
            elif occupy_pc > 0 and occupy_pc <= 49:
                lot_status = 1

        if facilityId == 2:  # MAC hypothetical lot capacity of 150
            capacity = 150
            occupy_pc = (carCount / capacity) * 100
            if occupy_pc > 90 and occupy_pc <= 100:
                lot_status = 7  # full
            elif occupy_pc > 70 and carCount <= 89:
                lot_status = 5  # avail fulling
            elif occupy_pc > 60 and occupy_pc <= 69:
                lot_status = 4
            elif occupy_pc > 50 and occupy_pc <= 59:
                lot_status = 3  # avail
            elif occupy_pc > 0 and occupy_pc <= 49:
                lot_status = 1

        if facilityId == 3:  # MCPN hypothetical lot capacity of 150
            capacity = 150
            occupy_pc = (carCount / capacity) * 100
            if occupy_pc > 90 and occupy_pc <= 100:
                lot_status = 7  # full
            elif occupy_pc > 70 and carCount <= 89:
                lot_status = 5  # avail fulling
            elif occupy_pc > 60 and occupy_pc <= 69:
                lot_status = 4
            elif occupy_pc > 50 and occupy_pc <= 59:
                lot_status = 3  # avail
            elif occupy_pc > 0 and occupy_pc <= 49:
                lot_status = 1

        if facilityId == 4:  # PepperGarageNW hypothetical lot capacity of 150
            capacity = 150
            occupy_pc = (carCount / capacity) * 100
            if occupy_pc > 90 and occupy_pc <= 100:
                lot_status = 7  # full
            elif occupy_pc > 70 and carCount <= 89:
                lot_status = 5  # avail fulling
            elif occupy_pc > 60 and occupy_pc <= 69:
                lot_status = 4
            elif occupy_pc > 50 and occupy_pc <= 59:
                lot_status = 3  # avail
            elif occupy_pc > 0 and occupy_pc <= 49:
                lot_status = 1
        return lot_status

        if facilityId == 5:  # PepperGarageSouth hypothetical lot capacity of 150
            capacity = 150
            occupy_pc = (carCount / capacity) * 100
            if occupy_pc > 90 and occupy_pc <= 100:
                lot_status = 7  # full
            elif occupy_pc > 70 and carCount <= 89:
                lot_status = 5  # avail fulling
            elif occupy_pc > 60 and occupy_pc <= 69:
                lot_status = 4
            elif occupy_pc > 50 and occupy_pc <= 59:
                lot_status = 3  # avail
            elif occupy_pc > 0 and occupy_pc <= 49:
                lot_status = 1

        if facilityId == 6:  # PepperN hypothetical lot capacity of 150
            capacity = 150
            occupy_pc = (carCount / capacity) * 100
            if occupy_pc > 90 and occupy_pc <= 100:
                lot_status = 7  # full
            elif occupy_pc > 70 and carCount <= 89:
                lot_status = 5  # avail fulling
            elif occupy_pc > 60 and occupy_pc <= 69:
                lot_status = 4
            elif occupy_pc > 50 and occupy_pc <= 59:
                lot_status = 3  # avail
            elif occupy_pc > 0 and occupy_pc <= 49:
                lot_status = 1

    def AddCount(self):
        # self.data = facilityCountOjbect
        (
            run_id,
            facility_id,
            raw_count,
            masked_count,
            image,
            payload,
            created_by,
            creation_date,
        ) = self

        # 1 - derive lot status
        # lot_status = (
        #     services.facilitycountsaveservice.FacilityCountSaveService.deriveLotStatus()
        # )
        # x - finally send to the repository

        facilityData = (
            run_id,
            facility_id,
            raw_count,
            masked_count,
            image,
            payload,
            created_by,
            creation_date,
        )
        repository.Insert(self)

        # bbox,label,conf,facelib = m.detect(image)
        # FacilityId, OccupiedSpots, ImageName, DataPayload, CreatedBy, CreationDate
        # facilityCountObject = (
        #     9,
        #     20,
        #     'lot status'
        #     "thisisapic.jpg",
        #     "{somejson{here}}",
        #     "PythonAgent",
        #     datetime.now(),
        # )


# conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server}; Server=10.12.13.16,1433; Database=ParkingAI;uid=parkingai_user;pwd=Videre2020!;TrustedConnect=True')
# cursor = conn.cursor()
# cursor.execute('SELECT * FROM ParkingAI.dbo.FacilityCounts')
# for row in cursor:
# print(row)
