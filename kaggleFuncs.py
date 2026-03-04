import kagglehub

def testingkaggleAPI():
    path = kagglehub.dataset_download("fmena14/crowd-counting")
    return path