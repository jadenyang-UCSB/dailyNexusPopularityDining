import os
import config
from bs4 import BeautifulSoup
import requests
from selenium.webdriver.common.by import By
from selenium import webdriver
import time
import numpy as np
from PIL import Image
import kagglehub


def reload(webChrome):
    time.sleep(1)
    webChrome.execute_script("location.reload()")
    time.sleep(1)