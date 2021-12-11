import pyodbc
from modules.conf import Conf
import modules.log as l

# logit()


class FacilityCountRepository:
    def Insert(facilityCountObject):
        (
            RunId,
            FacilityId,
            RawCount,
            MaskedCount,
            ImageName,
            DataPayload,
            CreatedBy,
            CreationDate,
        ) = facilityCountObject

        logit = l.Log()
        conf = Conf("config/appsettings.json")

        logit.info(
            f"Values to insert {RunId},{FacilityId},{RawCount},{MaskedCount},{ImageName},{DataPayload},{CreatedBy},{CreationDate}"
        )
        try:
            connection = pyodbc.connect(
                conf["dev_connection_string"]
                # conf["loco_connection_string"]
                # "Driver={ODBC Driver 17 for SQL Server}; Server=10.12.13.16,1433; Database=ParkingAI;uid=parkingai_user;pwd=Videre2020!;TrustedConnect=True"
            )
            cursor = connection.cursor()
            SQLCommand = "INSERT INTO FacilityCounts(RunId,FacilityId, RawCount,MaskedCount,ImageName, DataPayload, CreatedBy, CreationDate) VALUES (?,?,?,?,?,?,?,?)"
            # Processing Query
            cursor.execute(SQLCommand, facilityCountObject)
            # Commiting any pending transaction to the database.
            result = connection.commit()
            # closing connection
            devmsg = f"Data Successfully Inserted into {conf['dev_connection_string']}"
            logit.info(devmsg)
            connection.close()
        except pyodbc.Error as err:
            logit.info(err)
            result = err
        #PROD
        try:
            connection = pyodbc.connect(
                conf["prod_connection_string"]
                # "Driver={ODBC Driver 17 for SQL Server}; Server=10.12.13.16,1433; Database=ParkingAI;uid=parkingai_user;pwd=Videre2020!;TrustedConnect=True"
            )
            cursor = connection.cursor()
            SQLCommand = "INSERT INTO FacilityCounts(RunId,FacilityId, RawCount,MaskedCount,ImageName, DataPayload, CreatedBy, CreationDate) VALUES (?,?,?,?,?,?,?,?)"
            # Processing Query
            cursor.execute(SQLCommand, facilityCountObject)
            # Commiting any pending transaction to the database.
            result = connection.commit()
            # closing connection
            prodmsg = (
                f"Data Successfully Inserted into {conf['prod_connection_string']}"
            )
            logit.info(prodmsg)
            connection.close()
        except pyodbc.Error as err:
            logit.info(err)

        return result
