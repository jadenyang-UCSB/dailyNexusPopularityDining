import os
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import numpy as np
from PIL import Image
import kagglehub

def dictionaryData(csvPathway):
    dictNumber  = {}
    namesIndex = 0
    arrayPathway = pathwayAddition(csvPathway)
    with open (csvPathway,"r") as file:
        for line in file:
            original = line.strip()
            arrayString = original.split(",")
            if arrayString[0].isdigit():
                dictNumber[arrayPathway[namesIndex]] = int(arrayString[1])
                namesIndex = namesIndex + 1
            #Just know that the id,count for the labels.csv
    
    return dictNumber


def pathwayAddition(csvPathway):
    base = os.path.dirname(csvPathway)
    framePathway = os.path.join((os.path.join(base,"frames")),"frames")
    names = os.listdir(framePathway)
    names = sorted(names)
    return names