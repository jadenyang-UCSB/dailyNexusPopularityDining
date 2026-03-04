#Important libraries
import os
import config
import copy

#API Key Handle
from dotenv import load_dotenv


load_dotenv()
CARRILLO_URL = os.getenv("API_KEYCARRILLO")

if not CARRILLO_URL:
    raise RuntimeError("CARRILLO_URL is missing. Put it in .env")


# from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import numpy as np
# from PIL import Image
import kagglehub

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from ultralytics import YOLO
import cv2
# from scipy.optimize import linear_sum_assignment

import numpy as np
from matplotlib import pyplot as plts

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log", mode="a"),  # append
        logging.StreamHandler()                    # console
    ]
)


# Code from separate files
import seleniumFuncs
import scalar_utils

import osnet

class wrapper:
    def __init__(self):
        self.in_overflow = []
        self.inside = []
        self.out_overflow = []
        self.outside = []
        self.counted = []
        self.best_score = 0

    def cos01(self,a,b):
        c = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return (c + 1) / 2

    def push_value(self, value):
        # Default weight = 1 person unless specified (for wide/group boxes)
        if "weight" not in value:
            value["weight"] = 1
        # Re-ID match threshold (0–1). Slightly lower = more matches, more risk of merging two people.
        mean = 0.90
        pq = []
        # Weights tuned for 5-sec intervals: rely more on appearance (hist+ResNet), less on position.
        for i in self.counted:
            pos_sim = comparePosition(i["position"], value["position"])
            score = 0.85 * self.cos01(i["color"], value["color"]) + pos_sim * 0.15
            pq.append((score, i))
        
        if not pq:
            self.counted.append(value)
            return
        
        score, entry = max(pq, key=lambda t: t[0])
        

        if score >= mean:
            logging.info("Already in there")
            entry["color"] = value["color"]
            entry["croppedImage"] = value["croppedImage"]
            entry["time"] = 0
            # Keep the larger weight if we ever see this person as part of a wide/group box
            entry["weight"] = max(entry.get("weight", 1), value.get("weight", 1))
            # Use current zone (inside/outside polygon) for direction, not position delta.
            # direction True = in inside_poly, False = in outside_poly.
            entry["direction"] = value["direction"]
            entry["position"] = value["position"]
        else:
            self.counted.append(value)
        

def hsv_hist(region):
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 1], 15, 255)
    hist = cv2.calcHist([hsv], [0, 1], mask, [36, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten()

def removeBackground(img):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)


    h, w = img.shape[:2]
    rect = (1, 1, w, h)  

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img

def process_crop(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE(clipLimit= 2.0, tileGridSize=(4,4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def compareHistogram(image_A, image_B):
    if(image_B.shape[0] < 40):
        return 1.0

    a = process_crop(image_A)
    b = process_crop(image_B)

    height_a = a.shape[0]
    height_b = b.shape[0]
    mid_a = height_a//2
    mid_b = height_b//2

    upperA = image_A[:mid_a, :]
    lowerA = image_A[mid_a : , :]

    upperB = image_B[:mid_b, :]
    lowerB = image_B[mid_b : , :]

    upperHist = cv2.compareHist(hsv_hist(upperA), hsv_hist(upperB), cv2.HISTCMP_BHATTACHARYYA)
    lowerHist = cv2.compareHist(hsv_hist(lowerA), hsv_hist(lowerB), cv2.HISTCMP_BHATTACHARYYA)
    return upperHist * 0.5 + lowerHist * 0.5

def compareVector(vec_A, vec_B):
    if(vec_B is None):
        return False
    
    number = np.dot(vec_A,vec_B)
    magA = np.linalg.norm(vec_A)
    magB = np.linalg.norm(vec_B)
    denom = magA * magB
    if denom == 0:
        return 0.0
    overall = number/denom
    return overall

def comparePosition(pos_A, pos_B):
    if(pos_B is None):
        return 1.0
    distance = ((pos_A[0] - pos_B[0])**2 + (pos_A[1] - pos_B[1])**2)**0.5
    # Use 450px tolerance: with a frame every 5 sec, people can move a lot between frames.
    return max(0.0, 1.0 - distance / 450.0)

def click_event(event, x, y, flags, param):
    if(event == cv2.EVENT_LBUTTONDOWN):
        print(f"This is x: {x}. This is y: {y}.")

def inside_point(poly, p):
    return cv2.pointPolygonTest(poly,p,False) >= 0

def main():
    # These variables are mainly for tracking
    previous_frame = None
    MISSING_FRAMES_THRESHOLD = 3
    # This opens the google chrome
    webChrome = webdriver.Chrome()
    webChrome.get(CARRILLO_URL)

    #Important Paths:
    pathtoBlank = os.getenv("BLANKJPG")
    
    #AI Stuff
    model = YOLO("yolo11s.pt")
    
    flow = wrapper()

    messages = ["None", "Empty", "Quiet", "Moderate", "High", "Near Capacity"]
    current_idx = 1

    while True:
        # color_vector_list = []
        time.sleep(5)
        seleniumFuncs.reload(webChrome)
        
        for calibrate in flow.counted:
            calibrate["time"] += 1

        pictureWeb = webChrome.find_element(By.XPATH, "/html/body/img")
        originalPicture = pictureWeb.get_attribute("src")

        with open (pathtoBlank,"wb") as file:
            file.write(requests.get(originalPicture).content)


        # Coordinate finder
        npBlank = cv2.imread(pathtoBlank)

        #Ensures that it's a different picture each time
        if(previous_frame is not None):
            if(np.array_equal(previous_frame,npBlank)):
                continue
        previous_frame = npBlank.copy()
        tracking = model(pathtoBlank)
    
        #Detection stuff
        for b in tracking[0].boxes:
            
            if int(b.cls.item()) != 0 or float(b.conf) < 0.25 or b.conf == None:
                print(f"Skipping, {int(b.cls.item())} WITH A CONF OF {b.conf}")
                continue

            coordinate_point = []
            floats = b.xyxy[0].tolist()
            x1, y1, x2, y2 = floats
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            intx0,inty0,intx1,inty1 = map(int, floats)
            middle_X = (x2 + x1)/2
            middle_Y = (y2 + y1)/2
            coordinate_point.append(middle_X)
            coordinate_point.append(middle_Y) # base center point
            image = cv2.imread(pathtoBlank)
            
            cv2.imwrite(os.getenv("COPYJPG"),image)
            image = cv2.imread("./copy_image.jpg")
            
            cropImage = image[inty0:inty1,intx0:intx1]
            personVector = osnet.osnet_vector(cropImage)

            # Group handaling
            GROUP_MIN_WIDTH = 120.0 
            AVG_PERSON_WIDTH = 40.0 
            MAX_GROUP_SIZE = 5 

            if width >= GROUP_MIN_WIDTH:
                print("This is a group")
                est_group = max(1, int(round(width / AVG_PERSON_WIDTH)))
                group_weight = min(est_group, MAX_GROUP_SIZE)
            else:
                group_weight = 1
            
            inside_poly = np.array([
                [4,3],
                [438,5],
                [442,191],
                [5,494]
            ], dtype=np.int32)
            
            outside_poly = np.array([
                [377,570],
                [676,203],
                [824,284],
                [762,569]
            ], dtype=np.int32)

            # people inside or outside

            pos = [middle_X, middle_Y]
            base_value = {
                "color": personVector,
                "croppedImage": removeBackground(cropImage),
                "position": pos,
                "time": 0,
                "weight": group_weight,
            }

            if (inside_point(inside_poly, (intx0,inty0))):
                logging.info(f"We just initalized a person and we put them initally inside with a weight of {group_weight}")
                v = base_value.copy()
                v["direction"] = True
                flow.push_value(v)
            elif (inside_point(outside_poly, (intx1,inty1))):
                logging.info(f"We just initalized a person and we put them initally outside with a weight of {group_weight}")
                v = base_value.copy()
                v["direction"] = False
                flow.push_value(v)

        flow_remove = []

        for idx, timer in enumerate(flow.counted):
            missed_frames = scalar_utils.to_int(timer["time"])
            direction = scalar_utils.to_bool(timer["direction"])
            weight = timer.get("weight", 1)
            if missed_frames >= MISSING_FRAMES_THRESHOLD and direction:
                # Last seen in outside_poly (entrance) → treat as entered dining hall
                logging.info("Put it in flow.inside (entered)")
                for _ in range(weight):
                    flow.inside.append(timer)
                flow_remove.append(idx)
            elif missed_frames >= MISSING_FRAMES_THRESHOLD and not direction:
                logging.info("Put in flow.outside (exited)")
                for _ in range(weight):
                    flow.outside.append(timer)
                flow_remove.append(idx)
            elif missed_frames >= MISSING_FRAMES_THRESHOLD:
                flow_remove.append(idx)
        
        for item_remove in sorted(flow_remove, reverse=True):
            del flow.counted[item_remove]
        
        # Tune thresholds so the status moves more often; each ~15–20 people shifts a level
        INSIDE_STEP = 20
        OUTSIDE_STEP = 20
        if len(flow.inside) >= INSIDE_STEP:
            if(len(flow.out_overflow) != 0):
                flow.out_overflow.pop()
            else:
                current_idx += 1
            flow.inside = []

        if len(flow.outside) >= OUTSIDE_STEP:
            if(len(flow.in_overflow) != 0):
                flow.in_overflow.pop()
            else:
                current_idx -= 1
            flow.outside = []
        
        if(current_idx >= 5):
            flow.in_overflow.append("over")
            current_idx = 5
        elif(current_idx <= 0):
            flow.out_overflow.append("over")
            current_idx = 1

        logging.info(f"Flow Inside {len(flow.inside)}. Flow Outside: {len(flow.outside)}. Current overall: {messages[current_idx]}")
        for people in flow.counted:
            print(people["direction"])

        print("Inside: ", len(flow.inside))
        print("Outside: ", len(flow.outside))
        print(messages[current_idx])

    webChrome.close()


if __name__ == "__main__":
    main()