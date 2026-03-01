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
import random
# from scipy.optimize import linear_sum_assignment
import heapq

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


#Code from seperate files
import seleniumFuncs
import kaggleFuncs
import organizationData
import dataprep
import trainModel
import resNet


class wrapper:
    """Tracks people in zones and counts entries/exits.
    - counted: people currently being tracked (seen in inside_poly or outside_poly).
    - direction True = last seen in inside_poly, False = last seen in outside_poly.
    - When we stop re-identifying someone for MISSING_FRAMES_THRESHOLD frames,
      we count them: direction False → entered (flow.inside), direction True → exited (flow.outside).
    """
    def __init__(self):
        self.inside = []   # entered (last seen in outside/entrance zone)
        self.outside = []  # exited (last seen in inside/dining zone)
        self.counted = []
        self.best_score = 0

    def push_value(self, value):
        # Re-ID match threshold (0–1). Slightly lower = more matches, more risk of merging two people.
        mean = 0.55
        pq = []

        # Weights tuned for 5-sec intervals: rely more on appearance (hist+ResNet), less on position.
        for i in self.counted:
            x = compareHistogram(i["croppedImage"], value["croppedImage"])
            hist_sim = 1.0 - min(1.0, x)
            color_sim = compareVector(i["color"], value["color"])
            pos_sim = comparePosition(i["position"], value["position"])
            score = hist_sim * 0.5 + color_sim * 0.35 + pos_sim * 0.15
            pq.append((score, i))
        
        if not pq:
            self.counted.append(value)
            return
        
        score, entry = max(pq, key=lambda t: t[0])

        if(score >= mean):
            entry["color"] = value["color"]
            entry["croppedImage"] = value["croppedImage"]
            entry["time"] = 0
            # Use current zone (inside/outside polygon) for direction, not position delta.
            # direction True = in inside_poly, False = in outside_poly.
            entry["direction"] = value["direction"]
            entry["position"] = value["position"]
        else:
            print("New person, adding into the counted array")
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
        # assert img is not None, "file could not be read, check with os.path.exists()"

def process_crop(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    clahe = cv2.createCLAHE(clipLimit= 2.0, tileGridSize=(4,4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

def compareHistogram(image_A, image_B):
    if(image_B.shape[0] < 40):
        print("Shape is too small")
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
    # print(upperHist)
    lowerHist = cv2.compareHist(hsv_hist(lowerA), hsv_hist(lowerB), cv2.HISTCMP_BHATTACHARYYA)
    # print(lowerHist)
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
    # print(overall)
    
    # if(overall >= 0.66):
    #     return overall
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
    # Frames without re-ID before we count person as entered/exited (each frame = 5 sec image).
    # Lower = faster counting but more sensitive to missed detections. 3 = 15 sec.
    MISSING_FRAMES_THRESHOLD = 3
    # This opens the google chrome
    webChrome = webdriver.Chrome()
    webChrome.get(CARRILLO_URL)

    #This organizes the data. I used kaggle to get my datas
    if(os.path.exists("/Users/jadenyang/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3")):
        kaggle_data = "/Users/jadenyang/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3"
        csv_data = os.path.join(kaggle_data,"labels.csv")
    else:
        kaggle_data = kaggleFuncs.testingkaggleAPI()
        csv_data = os.path.join(kaggle_data,"labels.csv")

    #Organizes my data
    dictData = organizationData.dictionaryData(csv_data)
    array_pic, array_count = dataprep.resizeImage(dictData)

    #Important Paths:
    pathtoBlank = "./blank.jpeg"
    
    #Training
    if os.path.exists("./peopleCounter.keras"):
        people_counter_model = load_model("./peopleCounter.keras")
        history = None
    else:
        people_counter_model,history = trainModel.train(array_pic,array_count)

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
        # cv2.imshow("Image",npBlank)
        # cv2.setMouseCallback("Image", click_event)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        #Ensures that it's a different picture each time
        if(previous_frame is not None):
            if(np.array_equal(previous_frame,npBlank)):
                print("Skipping cause same picture")
                continue
        previous_frame = npBlank.copy()
        tracking = model(pathtoBlank)

        #Sanity Checker to ensure that the active number is counting
        data_fitting = load_img(pathtoBlank, target_size = (128,128))
        array_target = np.array(data_fitting)/255
        fittedData = np.expand_dims(array_target,axis=0)
        sanityCheck = people_counter_model.predict(fittedData)
        print(sanityCheck)
    
        #Detection stuff
        for b in tracking[0].boxes:
            
            if int(b.cls.item()) != 0 or float(b.conf) < 0.25 or b.conf == None:
                print(f"Skipping, {int(b.cls.item())} WITH A CONF OF {b.conf}")
                continue

            coordinate_point = []
            floats = b.xyxy[0].tolist()
            x1, y1, x2, y2 = floats
            
            intx0,inty0,intx1,inty1 = map(int, floats)
            middle_X = (x2 + x1)/2
            middle_Y = (y2 + y1)/2
            coordinate_point.append(middle_X)
            coordinate_point.append(middle_Y) #YOU HAVE COORDINATE POINT
            image = cv2.imread(pathtoBlank)
            
            cv2.imwrite("./copy_image.jpg",image)
            image = cv2.imread("./copy_image.jpg")
            
            cropImage = image[inty0:inty1,intx0:intx1]
            personVector = resNet.colorVector(cropImage) #YOU HAVE THE COLOR VECTOR OF THE PERSON
        


            inside_poly = np.array([
                [0,0],
                [178,569],
                [513,183],
                [638,6]
            ], dtype=np.int32)
            
            outside_poly = np.array([
                [400,570],
                [886,557],
                [955.,21],
                [890,2]
            ], dtype=np.int32)
            # debug_image = npBlank.copy()
            # cv2.polylines(debug_image, [inside_poly], True, (0, 255, 0), 2)    # green = inside
            # cv2.polylines(debug_image, [outside_poly], True, (0, 0, 255), 2)   # red = outsid
            # cv2.imshow("I hate this", debug_image)
            # cv2.waitKey(0)
            if(inside_point(inside_poly, (intx1,inty1)) or inside_point(inside_poly, coordinate_point) or inside_point(inside_poly, (intx0,inty0))):
                # logging.info("inside has been activated")
                flow.push_value(
                    {"color":personVector, 
                    "croppedImage": removeBackground(cropImage),
                    "position":coordinate_point, 
                    "time": 0, 
                    "direction": True}
                )
            elif(inside_point(outside_poly, (intx1,inty1)) or inside_point(outside_poly, coordinate_point) or inside_point(outside_poly, (intx0,inty0))):
                # logging.info("Outside has been activated")
                flow.push_value(
                    {"color":personVector, 
                    "croppedImage": removeBackground(cropImage),
                    "position":coordinate_point, 
                    "time": 0, 
                    "direction": False}
                )

        flow_remove = []
        for timer in flow.counted:
            if timer["time"] >= MISSING_FRAMES_THRESHOLD and timer["direction"] is False:
                # Last seen in outside_poly (entrance) → count as entered
                logging.info("Put it in flow.inside (entered)")
                flow.inside.append(timer)
                flow_remove.append(timer)
            elif timer["time"] >= MISSING_FRAMES_THRESHOLD and timer["direction"] is True:
                # Last seen in inside_poly (dining) → count as exited
                logging.info("Put in flow.outside (exited)")
                flow.outside.append(timer)
                flow_remove.append(timer)
            elif timer["time"] >= MISSING_FRAMES_THRESHOLD:
                flow_remove.append(timer)
        
        for item_remove in flow_remove:
            flow.counted.remove(item_remove)
        
        if(len(flow.inside) >= 40):
            current_idx += 1
            flow.inside = []
        if(len(flow.outside) >= 40):
            current_idx -= 1
            flow.outside = []
        
        if(current_idx >= 5):
            current_idx = 5
        elif(current_idx <= 0):
            current_idx = 1

        logging.info(f"Flow Inside {len(flow.inside)}. Flow Outside: {len(flow.outside)}. Current overall: {messages[current_idx]}")

        print("Inside: ", len(flow.inside))
        print("Outside: ", len(flow.outside))
        print(messages[current_idx])

    webChrome.close()



main()



# #Important libraries
# import os
# import config
# import copy

# #API Key Handle
# from dotenv import load_dotenv


# load_dotenv()
# CARRILLO_URL = os.getenv("API_KEYCARRILLO")

# if not CARRILLO_URL:
#     raise RuntimeError("CARRILLO_URL is missing. Put it in .env")


# # from bs4 import BeautifulSoup
# import requests
# from selenium.webdriver.common.by import By
# from selenium import webdriver
# import time
# import numpy as np
# # from PIL import Image
# import kagglehub
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import load_img, img_to_array
# from ultralytics import YOLO
# import cv2
# import random
# # from scipy.optimize import linear_sum_assignment
# import heapq

# import numpy as np
# from matplotlib import pyplot as plts

# import logging

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("app.log", mode="a"),  # append
#         logging.StreamHandler()                    # console
#     ]
# )


# #Code from seperate files
# import seleniumFuncs
# import kaggleFuncs
# import organizationData
# import dataprep
# import trainModel
# import resNet


# class wrapper:
#     """Tracks people in zones and counts entries/exits.
#     - counted: people currently being tracked (seen in inside_poly or outside_poly).
#     - direction True = last seen in inside_poly, False = last seen in outside_poly.
#     - When we stop re-identifying someone for MISSING_FRAMES_THRESHOLD frames,
#       we count them: direction False → entered (flow.inside), direction True → exited (flow.outside).
#     """
#     def __init__(self):
#         self.inside = []   # entered (last seen in outside/entrance zone)
#         self.outside = []  # exited (last seen in inside/dining zone)
#         self.counted = []
#         self.best_score = 0

#     def push_value(self, value):
#         # Re-ID match threshold (0–1). Slightly lower = more matches, more risk of merging two people.
#         mean = 0.55
#         pq = []

#         # Weights tuned for 5-sec intervals: rely more on appearance (hist+ResNet), less on position.
#         for i in self.counted:
#             x = compareHistogram(i["croppedImage"], value["croppedImage"])
#             hist_sim = 1.0 - min(1.0, x)
#             color_sim = compareVector(i["color"], value["color"])
#             pos_sim = comparePosition(i["position"], value["position"])
#             score = hist_sim * 0.5 + color_sim * 0.35 + pos_sim * 0.15
#             pq.append((score, i))
        
#         if not pq:
#             self.counted.append(value)
#             return
        
#         score, entry = max(pq, key=lambda t: t[0])

#         if(score >= mean):
#             entry["color"] = value["color"]
#             entry["croppedImage"] = value["croppedImage"]
#             entry["time"] = 0
#             # Use current zone (inside/outside polygon) for direction, not position delta.
#             # direction True = in inside_poly, False = in outside_poly.
#             entry["direction"] = value["direction"]
#             entry["position"] = value["position"]
#         else:
#             print("New person, adding into the counted array")
#             self.counted.append(value)
        

# def hsv_hist(region):
#     hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv[:, :, 1], 15, 255)
#     hist = cv2.calcHist([hsv], [0, 1], mask, [36, 32], [0, 180, 0, 256])
#     cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
#     return hist.flatten()

# def removeBackground(img):
#     mask = np.zeros(img.shape[:2],np.uint8)
#     bgdModel = np.zeros((1,65),np.float64)
#     fgdModel = np.zeros((1,65),np.float64)


#     h, w = img.shape[:2]
#     rect = (1, 1, w, h)  

#     cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
#     mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
#     img = img*mask2[:,:,np.newaxis]
#     return img
#         # assert img is not None, "file could not be read, check with os.path.exists()"

# def process_crop(image):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
#     clahe = cv2.createCLAHE(clipLimit= 2.0, tileGridSize=(4,4))
#     lab[:, :, 0] = clahe.apply(lab[:, :, 0])
#     return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

# def compareHistogram(image_A, image_B):
#     if(image_B.shape[0] < 40):
#         print("Shape is too small")
#         return 1.0

#     a = process_crop(image_A)
#     b = process_crop(image_B)

#     height_a = a.shape[0]
#     height_b = b.shape[0]
#     mid_a = height_a//2
#     mid_b = height_b//2

#     upperA = image_A[:mid_a, :]
#     lowerA = image_A[mid_a : , :]

#     upperB = image_B[:mid_b, :]
#     lowerB = image_B[mid_b : , :]

#     upperHist = cv2.compareHist(hsv_hist(upperA), hsv_hist(upperB), cv2.HISTCMP_BHATTACHARYYA)
#     # print(upperHist)
#     lowerHist = cv2.compareHist(hsv_hist(lowerA), hsv_hist(lowerB), cv2.HISTCMP_BHATTACHARYYA)
#     # print(lowerHist)
#     return upperHist * 0.5 + lowerHist * 0.5

# def compareVector(vec_A, vec_B):
#     if(vec_B is None):
#         return False
    
#     number = np.dot(vec_A,vec_B)
#     magA = np.linalg.norm(vec_A)
#     magB = np.linalg.norm(vec_B)
#     denom = magA * magB
#     if denom == 0:
#         return 0.0
#     overall = number/denom
#     # print(overall)
    
#     # if(overall >= 0.66):
#     #     return overall
#     return overall

# def comparePosition(pos_A, pos_B):
#     if(pos_B is None):
#         return 1.0
#     distance = ((pos_A[0] - pos_B[0])**2 + (pos_A[1] - pos_B[1])**2)**0.5
#     # Use 450px tolerance: with a frame every 5 sec, people can move a lot between frames.
#     return max(0.0, 1.0 - distance / 450.0)

# def click_event(event, x, y, flags, param):
#     if(event == cv2.EVENT_LBUTTONDOWN):
#         print(f"This is x: {x}. This is y: {y}.")

# def inside_point(poly, p):
#     return cv2.pointPolygonTest(poly,p,False) >= 0

# def main():
#     # These variables are mainly for tracking
#     previous_frame = None
#     # Frames without re-ID before we count person as entered/exited (each frame = 5 sec image).
#     # Lower = faster counting but more sensitive to missed detections. 3 = 15 sec.
#     MISSING_FRAMES_THRESHOLD = 3
#     # This opens the google chrome
#     webChrome = webdriver.Chrome()
#     webChrome.get(CARRILLO_URL)

#     #This organizes the data. I used kaggle to get my datas
#     if(os.path.exists("/Users/jadenyang/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3")):
#         kaggle_data = "/Users/jadenyang/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3"
#         csv_data = os.path.join(kaggle_data,"labels.csv")
#     else:
#         kaggle_data = kaggleFuncs.testingkaggleAPI()
#         csv_data = os.path.join(kaggle_data,"labels.csv")

#     #Organizes my data
#     dictData = organizationData.dictionaryData(csv_data)
#     array_pic, array_count = dataprep.resizeImage(dictData)

#     #Important Paths:
#     pathtoBlank = "./blank.jpeg"
    
#     #Training
#     if os.path.exists("./peopleCounter.keras"):
#         people_counter_model = load_model("./peopleCounter.keras")
#         history = None
#     else:
#         people_counter_model,history = trainModel.train(array_pic,array_count)

#     #AI Stuff
#     model = YOLO("yolo11s.pt")
    
#     flow = wrapper()

#     messages = ["None", "Empty", "Quiet", "Moderate", "High", "Near Capacity"]
#     current_idx = 1

#     while True:
#         # color_vector_list = []
#         time.sleep(5)
#         seleniumFuncs.reload(webChrome)
        
#         for calibrate in flow.counted:
#             calibrate["time"] += 1

#         pictureWeb = webChrome.find_element(By.XPATH, "/html/body/img")
#         originalPicture = pictureWeb.get_attribute("src")

#         with open (pathtoBlank,"wb") as file:
#             file.write(requests.get(originalPicture).content)


#         # Coordinate finder
#         npBlank = cv2.imread(pathtoBlank)
#         # cv2.imshow("Image",npBlank)
#         # cv2.setMouseCallback("Image", click_event)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows
#         #Ensures that it's a different picture each time
#         if(previous_frame is not None):
#             if(np.array_equal(previous_frame,npBlank)):
#                 print("Skipping cause same picture")
#                 continue
#         previous_frame = npBlank.copy()
#         tracking = model(pathtoBlank)

#         #Sanity Checker to ensure that the active number is counting
#         data_fitting = load_img(pathtoBlank, target_size = (128,128))
#         array_target = np.array(data_fitting)/255
#         fittedData = np.expand_dims(array_target,axis=0)
#         sanityCheck = people_counter_model.predict(fittedData)
#         print(sanityCheck)
    
#         #Detection stuff
#         for b in tracking[0].boxes:
            
#             if int(b.cls.item()) != 0 or float(b.conf) < 0.25 or b.conf == None:
#                 print(f"Skipping, {int(b.cls.item())} WITH A CONF OF {b.conf}")
#                 continue

#             coordinate_point = []
#             floats = b.xyxy[0].tolist()
#             x1, y1, x2, y2 = floats
            
#             intx0,inty0,intx1,inty1 = map(int, floats)
#             middle_X = (x2 + x1)/2
#             middle_Y = (y2 + y1)/2
#             coordinate_point.append(middle_X)
#             coordinate_point.append(middle_Y) #YOU HAVE COORDINATE POINT
#             image = cv2.imread(pathtoBlank)
            
#             cv2.imwrite("./copy_image.jpg",image)
#             image = cv2.imread("./copy_image.jpg")
            
#             cropImage = image[inty0:inty1,intx0:intx1]
#             personVector = resNet.colorVector(cropImage) #YOU HAVE THE COLOR VECTOR OF THE PERSON
        


#             inside_poly = np.array([
#                 [0,0],
#                 [178,569],
#                 [513,183],
#                 [638,6]
#             ], dtype=np.int32)
            
#             outside_poly = np.array([
#                 [400,570],
#                 [886,557],
#                 [955.,21],
#                 [890,2]
#             ], dtype=np.int32)
#             # debug_image = npBlank.copy()
#             # cv2.polylines(debug_image, [inside_poly], True, (0, 255, 0), 2)    # green = inside
#             # cv2.polylines(debug_image, [outside_poly], True, (0, 0, 255), 2)   # red = outsid
#             # cv2.imshow("I hate this", debug_image)
#             # cv2.waitKey(0)
#             if(inside_point(inside_poly, (intx1,inty1)) or inside_point(inside_poly, coordinate_point) or inside_point(inside_poly, (intx0,inty0))):
#                 # logging.info("inside has been activated")
#                 flow.push_value(
#                     {"color":personVector, 
#                     "croppedImage": removeBackground(cropImage),
#                     "position":coordinate_point, 
#                     "time": 0, 
#                     "direction": True}
#                 )
#             elif(inside_point(outside_poly, (intx1,inty1)) or inside_point(outside_poly, coordinate_point) or inside_point(outside_poly, (intx0,inty0))):
#                 # logging.info("Outside has been activated")
#                 flow.push_value(
#                     {"color":personVector, 
#                     "croppedImage": removeBackground(cropImage),
#                     "position":coordinate_point, 
#                     "time": 0, 
#                     "direction": False}
#                 )

#         flow_remove = []
#         for timer in flow.counted:
#             if timer["time"] >= MISSING_FRAMES_THRESHOLD and timer["direction"] is False:
#                 # Last seen in outside_poly (entrance) → count as entered
#                 logging.info("Put it in flow.inside (entered)")
#                 flow.inside.append(timer)
#                 flow_remove.append(timer)
#             elif timer["time"] >= MISSING_FRAMES_THRESHOLD and timer["direction"] is True:
#                 # Last seen in inside_poly (dining) → count as exited
#                 logging.info("Put in flow.outside (exited)")
#                 flow.outside.append(timer)
#                 flow_remove.append(timer)
#             elif timer["time"] >= MISSING_FRAMES_THRESHOLD:
#                 flow_remove.append(timer)
        
#         for item_remove in flow_remove:
#             flow.counted.remove(item_remove)
        
#         if(len(flow.inside) >= 40):
#             current_idx += 1
#             flow.inside = []
#         if(len(flow.outside) >= 40):
#             current_idx -= 1
#             flow.outside = []
        
#         if(current_idx >= 5):
#             current_idx = 5
#         elif(current_idx <= 0):
#             current_idx = 1

#         logging.info(f"Flow Inside {len(flow.inside)}. Flow Outside: {len(flow.outside)}. Current overall: {messages[current_idx]}")

#         print("Inside: ", len(flow.inside))
#         print("Outside: ", len(flow.outside))
#         print(messages[current_idx])

#     webChrome.close()



# main()


    # def push_out(self, value):
    #     for i in self.outside:
    #         if(compareHistogram(i["croppedImage"], value["croppedImage"]) and compareVector(i["color"], value["color"]) and comparePosition(i["position"], value["position"])):
    #             return False
    #     self.outside.append(value)
    #     return True


    # if(abs(y2-y1) > 50):

            #     inside_poly = np.array([
            #         [6,20],
            #         [407,21],
            #         [450,275],
            #         [97,568]
            #     ], dtype=np.int32)
                
            #     outside_poly = np.array([
            #         [227,555],
            #         [549,176],
            #         [800,232],
            #         [823,571]
            #     ], dtype=np.int32)

            #     # debug_image = npBlank.copy()
            #     # cv2.polylines(debug_image, [inside_poly], True, (0, 255, 0), 2)    # green = inside
            #     # cv2.polylines(debug_image, [outside_poly], True, (0, 0, 255), 2)   # red = outsid
            #     # cv2.imshow("I hate this", debug_image)
            #     # cv2.waitKey(0)

            #     if(inside_point(inside_poly, (intx1,inty1)) or inside_point(inside_poly, coordinate_point) or inside_point(inside_poly, (intx0,inty0))):
            #         print("inside has been activated")
            #         flow.push_in(
            #             {"color":personVector, 
            #             "croppedImage": removeBackground(cropImage),
            #             "position":coordinate_point, 
            #             "time": 0, 
            #             "direction": None}
            #         )
            #     elif(inside_point(outside_poly, (intx1,inty1)) or inside_point(outside_poly, coordinate_point) or inside_point(outside_poly, (intx0,inty0))):
            #         print("Outside has been activated")
            #         flow.push_out(
            #             {"color":personVector, 
            #             "croppedImage": removeBackground(cropImage),
            #             "position":coordinate_point, 
            #             "time": 0, 
            #             "direction": None}
            #         )
            # else:
            #     print("I am not pushing this because it's way too small with a size of ", abs(y2-y1))
                        #             {"color":personVector, 
            #             "croppedImage": removeBackground(cropImage),
            #             "position":coordinate_point, 
            #             "time": 0, 
            #             "direction": None}