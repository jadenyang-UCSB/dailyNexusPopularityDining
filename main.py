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

#Code from seperate files
import seleniumFuncs
import kaggleFuncs
import organizationData
import dataprep
import trainModel
import resNet

    
def compareVector(vec_A, vec_B):
    if(vec_B is None):
        return False
    
    number = np.dot(vec_A,vec_B)
    magA = np.linalg.norm(vec_A)
    magB = np.linalg.norm(vec_B)
    denom = magA * magB
    if denom == 0:
        print("This is always printing")
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
        return True, distance
    return False, distance

def fingerPrint(point, crop_upper,crop_lower):
    x1,y1,x2,y2 = point
    hsv = cv2.cvtColor(crop_upper,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:,:,1], 50, 255)
    upper_hist = cv2.calcHist([hsv],[0,1],mask,[180,64], [0,180, 0,256])
    cv2.normalize(upper_hist, upper_hist, 0, 1, cv2.NORM_MINMAX)

    hsv = cv2.cvtColor(crop_lower,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:,:,1], 50, 255)
    lower_hist = cv2.calcHist([hsv],[0,1],mask,[180,64], [0,180, 0,256])
    cv2.normalize(lower_hist, lower_hist, 0, 1, cv2.NORM_MINMAX)

    return {
    "upper": upper_hist,
    "lower": lower_hist,
    "height": (y2-y1)/2,
    "pos": ((x2+x1)/2,y2),
    "missing": 0,
    "previousPos": ((x2+x1)/2,y2),
    "id": None
    }

def calculateCost(point,crop):
    height = crop.shape[0]
    fp = fingerPrint(point,crop[height//2:,:],crop[:height//2,:])
    return fp

def inside_point(poly, p):
    return cv2.pointPolygonTest(poly,p,False) >= 0

def main():
    #These variables are mainly for tracking
    total_number = 0
    id_num = 0
    previous_frame = None
    MISSINGFRAME = 3
    #This opens the google chrome
    webChrome = webdriver.Chrome()
    webChrome.get(config.API_KEYORTEGA)

    previous_arr = []

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
    
    while True:
        current_arr = []

        seleniumFuncs.reload(webChrome)
        pictureWeb = webChrome.find_element(By.XPATH, "/html/body/img")
        originalPicture = pictureWeb.get_attribute("src")
        with open (pathtoBlank,"wb") as file:
            file.write(requests.get(originalPicture).content)

        npBlank = cv2.imread(pathtoBlank)
        cv2.destroyAllWindows()

        #Ensures that it's a different picture each time
        if(previous_frame is not None):
            if(np.array_equal(previous_frame,npBlank)):
                continue
        previous_frame = npBlank.copy()

        tracking = model(pathtoBlank)

        #Sanity Checker to ensure that the active number is counting
        data_fitting = load_img(pathtoBlank, target_size = (128,128))
        array_target = np.array(data_fitting)/255
        fittedData = np.expand_dims(array_target,axis=0)
        sanityCheck = people_counter_model.predict(fittedData)
        print(sanityCheck)


        image = cv2.imread(pathtoBlank)

        for b in tracking[0].boxes:
            if b.cls.item() != 0:
                continue
            #Setting the coordinate plane

            coordinate_point = []
            floats = b.xyxy[0].tolist()
            x1, y1, x2, y2 = floats
            
            intx0,inty0,intx1,inty1 = map(int, floats)
            middle_X = (x2 + x1)/2
            middle_Y = (y2 + y1)/2
            coordinate_point.append(middle_X)
            coordinate_point.append(middle_Y)
            croppedImage = image[inty0:inty1,intx0:intx1]
            hist = calculateCost((intx0,inty0,intx1,inty1),croppedImage)
            current_arr.append(hist)

        if len(previous_arr) == 0:
            for new_object in current_arr:
                new_object["id"] = id_num
                id_num += 1

            previous_arr = current_arr

            continue
        
        costMatrix = np.zeros((len(previous_arr),len(current_arr)),dtype= np.float32)

        for i, prev_obj in enumerate(previous_arr):
            for j, curr_obj in enumerate(current_arr):
                upper = cv2.compareHist(prev_obj["upper"],curr_obj["upper"],cv2.HISTCMP_BHATTACHARYYA)
                lower = cv2.compareHist(prev_obj["lower"], curr_obj["lower"],cv2.HISTCMP_BHATTACHARYYA)
                distance = np.array(prev_obj["pos"],dtype=np.float32) - np.array(curr_obj["pos"],dtype=np.float32)
                total_distance = np.linalg.norm(distance) / 500
                normal_distance = min(total_distance,1.0)
                totalCost = (upper*0.45) + (lower*0.45) + (normal_distance * 0.1)
                costMatrix[i][j] = totalCost


        
        row_cost, col_cost = linear_sum_assignment(costMatrix)

        matched_pairs = []
        matched_prev = []
        matched_curr = []
        removed = set()
        final_removed = []

        for prev_index,new_index in zip(row_cost,col_cost):
            pid = previous_arr[prev_index]["id"]
            if (costMatrix[prev_index][new_index] < 0.6):
                matched_prev.append(pid)
                matched_curr.append(current_arr[new_index]["id"])
                matched_pairs.append((prev_index,new_index,pid))

        dupe_entry = []

        for pairs in matched_pairs:
            old = pairs[0]
            new = pairs[1]
            pid = pairs[2]
            current_arr[new]["id"] = pid
            dupe_entry.append(pid)
            current_arr[new]["previousPos"] = previous_arr[old]["pos"]
            current_arr[new]["missing"] = 0

        for prev_object in previous_arr:
            pid = prev_object["id"]
            if pid not in matched_prev:
                prev_object["missing"] += 1
                if(prev_object["missing"] >= MISSINGFRAME):
                    removed.add(pid)
                    print("This is the previousPose", prev_object["previousPos"][1])
                    print("This is the position", prev_object["pos"][1])
                    if(prev_object["previousPos"][1] - prev_object["pos"][1]) > 0:
                        print("I added a number")
                        total_number += 1
                    else:
                        print("I subtracted one")
                        total_number -= 1
            elif pid in dupe_entry:
                removed.add(pid)
        


        for j, curr_object in enumerate(current_arr):
            if j not in matched_curr:
                curr_object["id"] = id_num
                id_num += 1

        for previous in previous_arr:
            if(previous["id"] not in dupe_entry or previous["id"] not in removed):
                final_removed.append(previous)
        
        for current in current_arr:
            final_removed.append(current)

        previous_arr = final_removed
        print(total_number)

    webChrome.close()


main()
