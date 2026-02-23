import os
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
#utils import load_img, img_to_array
def resizeImage(imageCount):
    arrayXPic = []
    arrayYCount = []
    framePath = "/Users/jadenyang/.cache/kagglehub/datasets/fmena14/crowd-counting/versions/3/frames/frames/"

    for i in imageCount.keys():
            if (i != ".DS_Store"):
                imageTemp = load_img(os.path.join(framePath,i),target_size = (128,128))
                imageArray = np.array(img_to_array(imageTemp))/255
                arrayXPic.append(imageArray)
                arrayYCount.append(imageCount[i])
        
    # print(arrayYCount)
    return np.array(arrayXPic), np.array(arrayYCount,dtype=float)
