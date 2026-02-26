#Important libraries
import os
import config
import copy
# from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import numpy as np
from PIL import Image
import kagglehub
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from ultralytics import YOLO
import cv2
import random
from scipy.optimize import linear_sum_assignment

import numpy as np
from matplotlib import pyplot as plts


#Code from seperate files
import seleniumFuncs
import kaggleFuncs
import organizationData
import dataprep
import trainModel
import resNet


class wrapper:
    def __init__(self):
        self.inside = []
        self.outside = []
    
    def push_in(self, value):
        for i in self.inside:
            if(compareHistogram(i["croppedImage"], value["croppedImage"]) and compareVector(i["color"], value["color"]) and comparePosition(i["position"], value["position"])):
                print("Already exists")
                return False
        self.inside.append(value)
        print("New person")
        return True


    def push_out(self, value):
        for i in self.outside:
            if(compareVector(i["color"], value["color"]) and comparePosition(i["color"], value["color"])):
                return False
        self.outside.append(value)
        return True

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
    if(image_A.shape[0] < 90 or image_B.shape[0] < 90):
        print("Shape is too small")
        return False

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
    print(upperHist)
    lowerHist = cv2.compareHist(hsv_hist(lowerA), hsv_hist(lowerB), cv2.HISTCMP_BHATTACHARYYA)
    print(lowerHist)
    return upperHist < 0.6 and lowerHist < 0.6

def compareVector(vec_A, vec_B):
    if(vec_B is None):
        return False
    
    number = np.dot(vec_A,vec_B)
    magA = np.linalg.norm(vec_A)
    magB = np.linalg.norm(vec_B)
    denom = magA * magB
    if denom == 0:
        return False
    overall = number/denom
    # print(overall)
    
    if(overall >= 0.66):
        return True
    return False

def comparePosition(pos_A, pos_B):
    if(pos_B is None):
        return False
    distance = ((pos_A[0] - pos_B[0])**2 + (pos_A[1] - pos_B[1])**2)**0.5
    
    if(distance < 200):
        return True
    
    return False


def main():
    #These variables are mainly for tracking
    # total_number = 0
    # id_num = 0
    previous_frame = None
    MISSINGFRAME = 3
    #This opens the google chrome
    webChrome = webdriver.Chrome()
    webChrome.get(config.API_KEYDLG)

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

    while True:
        # color_vector_list = []
        time.sleep(5)
        seleniumFuncs.reload(webChrome)
        
        pictureWeb = webChrome.find_element(By.XPATH, "/html/body/img")
        originalPicture = pictureWeb.get_attribute("src")

        with open (pathtoBlank,"wb") as file:
            file.write(requests.get(originalPicture).content)
        npBlank = cv2.imread(pathtoBlank)

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
            
            if b.cls.item() != 0 or float(b.conf) < 0.5 or b.conf == None:
                print("Skipping")
                continue
            else:
                print("It's fine let's continue", float(b.conf))
            #Setting the coordinate plane
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
            if(abs(y2-y1) > 90):
                flow.push_in(
                    {"color":personVector, 
                    "croppedImage": removeBackground(cropImage),
                    "position":coordinate_point, 
                    "time": 0, 
                    "direction": None}
                )
            else:
                print("I am not pushing this because it's way too small with a size of ", abs(y2-y1))
            print('HOW MANY UNIQUE PPL', len(flow.inside))

    webChrome.close()



main()


            # for p in vector_inside:
            #     if compareVector(p, personVector):
            #         continue
# def fingerPrint(point, crop_upper,crop_lower):
#     x1,y1,x2,y2 = point
#     hsv = cv2.cvtColor(crop_upper,cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv[:,:,1], 50, 255)
#     upper_hist = cv2.calcHist([hsv],[0,1],mask,[180,64], [0,180, 0,256])
#     cv2.normalize(upper_hist, upper_hist, 0, 1, cv2.NORM_MINMAX)

#     hsv = cv2.cvtColor(crop_lower,cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv[:,:,1], 50, 255)
#     lower_hist = cv2.calcHist([hsv],[0,1],mask,[180,64], [0,180, 0,256])
#     cv2.normalize(lower_hist, lower_hist, 0, 1, cv2.NORM_MINMAX)

#     return {
#     "upper": upper_hist,
#     "lower": lower_hist,
#     "height": (y2-y1)/2,
#     "pos": ((x2+x1)/2,y2),
#     "missing": 0,
#     "previousPos": ((x2+x1)/2,y2),
#     "id": None
#     }

# def calculateCost(point,crop):
#     height = crop.shape[0]
#     fp = fingerPrint(point,crop[height//2:,:],crop[:height//2,:])
#     return fp


#    while True:

#         entering_interval = []
#         exiting_interval = []

#         vector_inside = []

#         enter_people = 0
#         exit_people = 0

#         overall = 0
#         seleniumFuncs.reload(webChrome)
#         pictureWeb = webChrome.find_element(By.XPATH, "/html/body/img")
#         originalPicture = pictureWeb.get_attribute("src")
#         with open (pathtoBlank,"wb") as file:
#             file.write(requests.get(originalPicture).content)

#         npBlank = cv2.imread(pathtoBlank)
#         cv2.destroyAllWindows()

#         #Ensures that it's a different picture each time
#         if(previous_frame is not None):
#             if(np.array_equal(previous_frame,npBlank)):
#                 continue
#         previous_frame = npBlank.copy()

#         tracking = model(pathtoBlank)

#         #Sanity Checker to ensure that the active number is counting
#         data_fitting = load_img(pathtoBlank, target_size = (128,128))
#         array_target = np.array(data_fitting)/255
#         fittedData = np.expand_dims(array_target,axis=0)
#         sanityCheck = people_counter_model.predict(fittedData)
#         print(sanityCheck)


#         image = cv2.imread(pathtoBlank)

# def inside_point(poly, p):
#     return cv2.pointPolygonTest(poly,p,False) >= 0