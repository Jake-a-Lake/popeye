# USAGE
# python3 cars.py --model frcnn-resnet --image images/Frame_2021-08-12_10-40-42.jpg --labels coco_classes.pickle
# NOTE: Models this detection script works with several models which will be downloaded on first use
# 1. Faster R-CNN with a ResNet50 backbone - Slower-More Accurate 		- frcnn-resnet
# 2. Faster R-CNN with a MobileNet v3 backbone - Faster-Less Accurate	- frcnn-mobilenet
# 3. RetinaNet with ResNet50 backbone - Balance of Speed and Accuracy 	- retinanet

# import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
import os
import platform
import glob
import time
from datetime import datetime, timedelta
from dateutil.tz import tzutc, tzlocal
import pytz
from dateutil import parser as dps
import pathlib
import imutils
from services.facilitycountsaveservice import FacilityCountSaveService
from models.facility import Facility
from modules.conf import Conf
import schedule
import time
import uuid
import socket

# import modules.common_params as g
import modules.log as l

conf = Conf("config/appsettings.json")
AGENT_NAME = f"{conf['agentname']}-{socket.gethostname()}"

# logit()
logit = l.Log()
if conf["debug"]:
    ROOT_DIR = "/mnt/image_storage"
else:
    ROOT_DIR = conf["root_directory"]

SUB_DIR = conf["subdir"]

logit.info(
    f"Starting up rainman from {AGENT_NAME} - we're counting cars: in the dir {ROOT_DIR}."
)
logit.info(
    f"If you are running locally in a container, you will most likely want to be reading out of /mnt/v/ rather than /image_storage/"
)
logit.info("Here are the directories available to be read:")
logit.info(
    "----------------------------------------------------------------------------"
)
# list directories for target root so we know if we've mounted the drive succesfully
# for x in os.listdir(ROOT_DIR):
#     print(x)
# logit(filter(lambda x: os.path.isdir(x), os.listdir(ROOT_DIR)))
for x in os.listdir(ROOT_DIR):
    if os.path.isfile(x):
        logit.info(f"f-{x}")
    elif os.path.isdir(x):
        logit.info(f"d-{x}")
    elif os.path.islink(x):
        logit.info(f"l-{x}")
    else:
        logit.info(f"---{x}")
logit.info(
    "----------------------------------------------------------------------------"
)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-c",
    "--config",
    required=True,
    help="path to desired configuration file (appsettings.json)",
)

# ap.add_argument("-i", "--image", type=str, required=False,
# 	help="path to the input image")
# ap.add_argument("-d", "--dir", type=str, required=False, help="directory to iterate")
# ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
# 	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
# 	help="name of the object detection model")
# ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
# 	help="path to file containing list of categories in COCO dataset")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-df", "--datefilter", type=str, default= datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d'),
# 	help="process files created after given date")

# args = vars(ap.parse_args())
# conf = imutils.Conf(args["config"])
# ===== ⏲ Get a list of timezones! ⏲
# python3
# import pytz
# pytz.all_timezone
LOCAL_TIMEZONE = pytz.timezone("US/Arizona")

MODE = conf["mode"]
MODEL_NAME = conf["model"]
CONFIDENCE_THRESHOLD = conf["confidence"]
DATE_FILTER = conf["datefilter"]
LABELS = conf["labels"]
TARGET_DIRS = conf["directories"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACILITY_ID = 0
# if args['image']:
# global name of file we are analyzing
# SOURCE_IMAGE=os.path.basename(args["image"])
# ANALYZED_IMAGE= f"{SOURCE_IMAGE.replace('.jpg','')}_{MODEL}.png"
# logit.info("SOURCE IMAGE {}".format(SOURCE_IMAGE))
# logit.info("ANALYZED IMAGE {}".format(ANALYZED_IMAGE))
# if args['dir']:
# define target direcotry from arguments
# set the device we will be using to run the model

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open(LABELS, "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and it's corresponding
# torchvision function call
MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn,
}

# load the model and set it to evaluation mode
model = MODELS[conf["model"]](
    pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True
).to(DEVICE)
model.eval()


def get_creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == "Windows":
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def get_newest_file(path):
    # files = os.listdir(path)
    files = glob.iglob(f"{path}/*.jpg")
    paths = [os.path.join(path, basename) for basename in files]
    if len(paths) > 0:
        newest = max(paths, key=os.path.getctime)
        return newest
    else:
        return ""


def analyzeImage(run_id, facility_id, filepath, file, dir):
    RUN_ID = run_id
    FACILITY_ID = facility_id
    ANALYZED_IMAGE = f"{file.replace('.jpg','')}_{MODEL_NAME}.png"
    # set up our counters for logging
    RAW_COUNT = 0
    MASKED_COUNT = 0
    points = np.array([])
    logit.info("SOURCE IMAGE {}".format(file))
    logit.info(f"Analyzing {ANALYZED_IMAGE} for facilityid: {FACILITY_ID}")

    # load the image from disk
    image = cv2.imread(filepath)
    orig = image.copy()
    height, width, channels = orig.shape

    # slice mask to ROI ------------------------------CHECK FOR LOTS THAT NEED MASKING HERE-------------------
    if FACILITY_ID in [2, 10, 11, 12]:  # 2 = MAC, 10,11,12 = GOLD
        # mask = mask[200:480, 0:640]
        # #cv2.imshow("Mask", mask)
        # cv2.imwrite("mask.png",mask)

        # create a black background image of the same shape of original using np zeroes
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)

        ###draw our polygon mask from the points gathered depending on the file origin
        if FACILITY_ID == 2:  # mac
            points = np.array(
                [
                    [
                        [7, 607],
                        [1371, 385],
                        [1919, 468],
                        [1915, 1027],
                        [1200, 799],
                        [385, 1077],
                        [1, 1077],
                    ]
                ]
            )
            cv2.drawContours(
                mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA
            )  # create a smooth region
            masked_image = cv2.bitwise_and(image, image, mask=mask)

        if FACILITY_ID == 10:  # gold212
            points = np.array(
                [[[472, 604], [1285, 523], [1917, 707], [1287, 1064], [456, 711]]]
            )
            cv2.drawContours(
                mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA
            )  # create a smooth region
            masked_image = cv2.bitwise_and(image, image, mask=mask)

        if FACILITY_ID == 11:  # gold214
            points = np.array(
                [
                    [
                        [583, 391],
                        [1, 579],
                        [159, 1077],
                        [1916, 1076],
                        [1868, 525],
                        [717, 380],
                    ]
                ]
            )
            cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

        if FACILITY_ID == 12:  # gold2116
            points = np.array([[[1915, 321], [1724, 277], [1748, 708], [1916, 721]]])
            cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
            masked_image = cv2.bitwise_and(image, image, mask=mask)

        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        masked_image = masked_image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        masked_image = np.expand_dims(masked_image, axis=0)
        masked_image = masked_image / 255.0
        masked_image = torch.FloatTensor(masked_image)

        # send the input to the device and pass the it through the network to
        # get the detections and predictions
        masked_image = masked_image.to(DEVICE)
        masked_detections = model(masked_image)[0]
        # loop over the detections for the masked image
        for i in range(0, len(masked_detections["boxes"])):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = masked_detections["scores"][i]
            idx = int(masked_detections["labels"][i])
            CLASS = CLASSES[idx]
            # logit.info("TAG: {}").format(idx)
            # logit.info(CLASS)
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence and is in some sort of vehicle class
            if (
                confidence > CONFIDENCE_THRESHOLD
                and CLASS == "car"
                or idx == "truck"
                or idx == "motorcycle"
                or idx == "trailer"
                or idx == "bus"
            ):  # add criterion for cars here
                MASKED_COUNT += 1
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                box = masked_detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction to our terminal
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                logit.info("{}".format(label))

                # define bgr colors for boundingbox and labels
                RED = (53, 32, 188)
                YELLOW = (30, 126, 226)
                GREEN = (9, 150, 37)
                # draw the bounding box and label on the image
                if confidence >= 0.70:
                    cv2.rectangle(orig, (startX, startY), (endX, endY), GREEN, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        orig,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        GREEN,
                        2,
                    )
                elif confidence <= 0.69 and confidence >= 0.30:
                    cv2.rectangle(orig, (startX, startY), (endX, endY), YELLOW, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        orig,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        YELLOW,
                        2,
                    )
                else:
                    cv2.rectangle(orig, (startX, startY), (endX, endY), RED, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(
                        orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2
                    )

    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)

    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    image = image.to(DEVICE)
    raw_detections = model(image)[0]

    # loop over the detections for the raw image
    for i in range(0, len(raw_detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = raw_detections["scores"][i]
        idx = int(raw_detections["labels"][i])
        CLASS = CLASSES[idx]
        # logit.info("TAG: {}").format(idx)
        # logit.info(CLASS)
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence and is in some sort of vehicle class
        if (
            confidence > CONFIDENCE_THRESHOLD
            and CLASS == "car"
            or idx == "truck"
            or idx == "motorcycle"
            or idx == "trailer"
            or idx == "bus"
        ):  # add criterion for cars here
            RAW_COUNT += 1
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            box = raw_detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            logit.info("{}".format(label))

            # define bgr colors for boundingbox and labels
            RED = (53, 32, 188)
            YELLOW = (30, 126, 226)
            GREEN = (9, 150, 37)
            # draw the bounding box and label on the image
            if confidence >= 0.70:
                cv2.rectangle(orig, (startX, startY), (endX, endY), GREEN, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2
                )
            elif confidence <= 0.69 and confidence >= 0.30:
                cv2.rectangle(orig, (startX, startY), (endX, endY), YELLOW, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2
                )
            else:
                cv2.rectangle(orig, (startX, startY), (endX, endY), RED, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(
                    orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2
                )

    # We're moving derivatives like this to a view in the db
    # LOT_STATUS = FacilityCountSaveService.deriveLotStatus(FACILITY_ID, RAW_COUNT)
    # MASKED_COUNT = RAW_COUNT

    file_ctime = get_creation_date(filepath)
    FILE_CTIME = dps.parse(time.ctime(file_ctime), fuzzy=True)
    LOCAL_FCTIME = FILE_CTIME.astimezone(LOCAL_TIMEZONE)
    facilityCountObject = (
        RUN_ID,
        FACILITY_ID,
        RAW_COUNT,
        MASKED_COUNT,
        ANALYZED_IMAGE,
        "",
        AGENT_NAME,
        LOCAL_FCTIME,
    )

    # Add summary to the image TODO: Detect pixel lightness and determine label should be dark mode!
    msg = f"RAW VEHICLE COUNT: {RAW_COUNT}  MASKED VEHICLE COUNT: {MASKED_COUNT}"
    logit.info(RAW_COUNT)
    center = int(width / 2)
    logit.info(center)
    cv2.putText(
        orig, msg, (int(width / 2), 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2
    )
    cv2.putText(
        orig,
        msg,
        (int(width / 2), 1060),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    # TODO: Draw the mask on the source image if it exists! DONE
    if points.size != 0:
        cv2.polylines(orig, [points], True, (0, 255, 0), thickness=3)

    ### show the output image ###
    # cv2.imshow("Output", orig)
    # cv2.waitKey(0)
    # new_file = os.path.join("analyzed/",os.path.basename(orig))
    # old_file = os.path.splitext(orig)[0]
    # logit.info(old_file)

    # NEW_PATH=os.path.join(os.getcwd(),'mnt','f','TEMP','Analyzed',ANALYZED_IMAGE)
    NEW_PATH = os.path.join(dir, "Analyzed", ANALYZED_IMAGE)
    logit.info(f"NEW PATH: {NEW_PATH}")
    cv2.imwrite(NEW_PATH, orig)

    # ------------------------------------------- send to the database!-----------------------------------------------
    FacilityCountSaveService.AddCount(facilityCountObject)

    # move file to analyzed folder to reduce load on iteration
    SOURCE_IMG_NEW = os.path.join(dir, "Analyzed", f"{file.replace('.jpg','')}.png")
    try:
        os.replace(filepath, SOURCE_IMG_NEW)
    except OSError as ex:
        exc_info = ex
        logit.error(f"There was an error saving analyzed file: {exc_info}")

    msg = f"Moving source file {filepath} to Analyzed Folder"
    logit.info(msg)


def countcars():
    # -----------------------------PROCESS BEGINS HERE ----------------------------------------
    RUN_ID = uuid.uuid4()  # create a run id for this
    logit.info(f"New Run Started - {RUN_ID}")
    for tar_dir in TARGET_DIRS:
        # enumval = Facility["COMEmployee"]  #testing enums
        enum_member = Facility[f"{tar_dir}"]
        FACILITY_ID = enum_member.value
        # target_dir = f"/mnt/v/{tar_dir}/grabs/"
        target_dir = f"{ROOT_DIR}/{tar_dir}/{SUB_DIR}"
        # ------------------------------OPERATE BY MODE IN CONFIG FILE-------------------
        if MODE == "scan":
            for root, dirs, files in os.walk(target_dir):
                for dir in dirs:
                    for file in files:
                        filepath = os.path.join(target_dir, file)
                        if filepath.find(f"{conf['output_folder']}") == -1:
                            createdtime = get_creation_date(filepath)
                            ctime = dps.parse(time.ctime(createdtime), fuzzy=True)
                            # modifiedtime= os.path(gtmtime(file))
                            # datefilter = datetime.strptime(DATE_FILTER,'%mm %dd %Y')
                            datefilter = dps.parse(DATE_FILTER, fuzzy=True)
                            if file.endswith(".jpg") and ctime > datefilter:
                                logit.info(
                                    f"{target_dir} matches criterion {datefilter}"
                                )
                                # ------analyze the image!---------------------
                                analyzeImage(RUN_ID, FACILITY_ID, filepath, file, root)
                            else:
                                logit.info(
                                    f"{filepath} does not match criterion {datefilter}."
                                )
        elif MODE == "latest":
            newest_path = get_newest_file(target_dir)
            if newest_path.endswith(".jpg") and newest_path.find("Analyzed") == -1:
                logit.info(f"Working with latest file in {tar_dir}: {target_dir}")
                # ------analyze the image!---------------------
                analyzeImage(
                    RUN_ID, newest_path, os.path.basename(newest_path), target_dir
                )
            else:
                logit.info(f"Could not identify latest file in {tar_dir} ")


# rt = RepeatedTimer(5, countcars, "World")  # it auto-starts, no need of rt.start()
# try:
#     sleep(5)  # your long-running job goes here...
# finally:
#     rt.stop()  # better in a try/finally block to make sure the program ends!
if conf["debug"] == True:
    countcars()
else:
    # https://schedule.readthedocs.io/en/stable/
    id = uuid.uuid4()
    schedule.every(5).minutes.do(countcars)
    while True:
        schedule.run_pending()
        time.sleep(1)
