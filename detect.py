import torch
import torch.nn.functional as F
from torchvision.ops import nms
from net12FCN import NetFCN
from net24 import Net as Net24
import cv2 as cv
import numpy as np
import imutils
import os


def overlay(image, bounding_boxes):
    bounding_boxes = np.array(bounding_boxes, dtype=np.uint32)
    for x1, y1, x2, y2 in bounding_boxes:
        cv.rectangle(image, (y1, x1), (y2, x2), (0, 255, 0), 2)
    cv.imshow('image', image)
    cv.waitKey(0)


def evaluate_model(path, detect, suffix='.jpg'):
    f = open(path, 'r')
    out = open("fold-01-out.txt", 'w')
    for line in f:
        assert os.path.isfile(os.path.join('data/fddb/images', line[:-1] + suffix))
        image = cv.imread(os.path.join('data/fddb/images', line[:-1] + suffix))
        boxes, scores = detect.detect(image)
        write_to_file(out, boxes, scores, line[:-1])


def write_to_file(file, boxes, scores, image):
    file.write(image + "\n")
    file.write(str(len(boxes)) + "\n")
    for i in range(len(boxes)):
        H = W = boxes[i][2] - boxes[i][0]
        y = max(0, boxes[i][1] - 0.1 * H)
        x = boxes[i][0]
        H *= 1.2
        file.write(str(y) + " " + str(x) + " " + str(H) + " " + str(H) + " " + str(scores[i]) + "\n")


def check_24net(boxes, image, model):
    idx = []
    with torch.no_grad():
        for box in boxes:
            x_l, y_l, _, _ = box
            crop = image[0:, 0:, x_l:x_l + 12, y_l:y_l + 12]
            output = model(F.interpolate(crop, size=24).float()).squeeze()
            # idx.append(torch.softmax(output, dim=0).numpy()[0] > 0.5)
            idx.append(output[0] > output[1])
    return idx


class Detect:

    def __init__(self, model12, scales, iou_th, model24=None):
        self.model12 = model12
        self.model12.eval()
        self.scales = scales
        self.iou_th = iou_th
        self.model24 = model24
        if model24:
            self.model24.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, image):
        """
        args
            - image (numpy.ndarray) in channels_last format
        returns
            - list of bounding boxes
        """
        bounding_boxes = []
        scores = []
        for scale in self.scales:
            w = int(image.shape[1] * scale)
            h = int(image.shape[0] * scale)
            if w <= 1 or h <= 1:
                continue
            scaled_image = imutils.resize(image, width=w, height=h)
            if scaled_image.shape[0] < 12 or scaled_image.shape[1] < 12:
                continue
            scaled_image = np.rollaxis(scaled_image, 2, 0)  # convert to channels_first
            scaled_image = torch.tensor(scaled_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model12(scaled_image.float()).squeeze(0)
            probability_map = torch.softmax(output, dim=1).numpy()[0]
            x, y = np.indices(output.shape[1:])
            positive_idx = probability_map > 0.5
            x, y = 2 * x[positive_idx], 2 * y[positive_idx]
            probability_map = probability_map[positive_idx]
            boxes_for_nms = np.rollaxis(np.stack([x, y, x + 12, y + 12]), 1, 0)
            if self.model24:
                idx = check_24net(boxes_for_nms, scaled_image, self.model24)
                probability_map = probability_map[idx]
                boxes_for_nms = boxes_for_nms[idx]
                scores.extend(probability_map)
                bounding_boxes.extend(boxes_for_nms / scale)
            else:
                boxes_for_nms = torch.tensor(boxes_for_nms).float().to(self.device)
                probability_map = torch.tensor(probability_map).float().to(self.device)
                boxes_after_nms_idx = nms(boxes_for_nms, probability_map, self.iou_th)
                b = boxes_for_nms[boxes_after_nms_idx].numpy()
                scores.extend(probability_map[boxes_after_nms_idx].numpy())
                bounding_boxes.extend(b / scale)
        if self.model24:
            scores = torch.tensor(scores).float()
            bounding_boxes = torch.tensor(bounding_boxes).float()
            boxes_after_nms_idx = nms(bounding_boxes, scores, self.iou_th)
            bounding_boxes = bounding_boxes[boxes_after_nms_idx].numpy().squeeze()
            scores = scores[boxes_after_nms_idx].numpy()
        return bounding_boxes, scores


model12 = NetFCN()
model24 = Net24()

state_dict_12 = torch.load('12FCN_2300.pt', map_location=torch.device('cpu'))['model_state_dict']
state_dict_24 = torch.load('24net_820.pt', map_location=torch.device('cpu'))['model_state_dict']

model12.load_state_dict(state_dict_12)
model24.load_state_dict(state_dict_24)
scales = [0.05, 0.08, 0.13, 0.2]  # [0.08 + 0.02 * i for i in range(10)]
detect = Detect(model12, scales=scales, iou_th=0.5)  # , model24=model24)
# image = cv.imread('img.jpg')
# boxes, scores = detect.detect(image)
# overlay(image, boxes)
evaluate_model("data/fddb/FDDB-folds/FDDB-fold-01.txt", detect)
