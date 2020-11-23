import torch
from net12FCN import NetFCN
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield x, y, image[y:y + windowSize[1], x:x + windowSize[0]]


model = NetFCN()
state_dict = torch.load('12NetFCN.pt', map_location=torch.device('cpu'))['model_state_dict']
model.load_state_dict(state_dict)
model.eval()
image = cv.imread('img.jpg')
batch_size = 256
windows = []
x_coords = []
y_coords = []
for (x, y, window) in sliding_window(image, stepSize=4, windowSize=(12, 12)):
    # if the window does not meet our desired window size, ignore it
    if window.shape[0] != 12 or window.shape[1] != 12:
        continue
    windows.append((np.rollaxis(image, 2, 0)))  # convert to channels_first
    x_coords.append(x)
    y_coords.append(y)

batches = np.array_split(windows, int(len(windows) / batch_size) + 1)
predictions = []
for batch in batches:
    with torch.no_grad:
        outputs = model(batch)
    probs = torch.softmax(outputs, dim=1)
