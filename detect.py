import torch
from torchvision.ops import nms
from net12FCN import NetFCN
import cv2 as cv
import numpy as np
import imutils
import os


def overlay(image, bounding_boxes):
    for x1, y1, x2, y2 in bounding_boxes:
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def evaluate_model(path, detect, suffix='.jpg'):
    f = open(path, 'r')
    out = open("12-net_results/fold-01-out.txt", 'w')
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
        x = max(0, boxes[i][0] - 0.1 * H)
        y = boxes[i][1]
        H *= 1.2
        file.write(str(x) + " " + str(y) + " " + str(W) + " " + str(H) + " " + str(scores[i]) + "\n")


class Detect:

    def __init__(self, model, scale, min_size, iou_th):
        self.model = model
        self.model.eval()
        self.scale = scale
        self.min_size = min_size
        self.iou_th = iou_th
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
        for i, scaled_image in enumerate(self.pyramid(image)):
            scaled_image = np.rollaxis(scaled_image, 2, 0)  # convert to channels_first
            scaled_image = torch.tensor(scaled_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(scaled_image.float()).squeeze()
            probability_map = torch.softmax(output, dim=1).numpy()[0]
            x, y = np.indices(output.shape[1:])
            positive_idx = probability_map > 0.5
            x, y = 2 * x[positive_idx], 2 * y[positive_idx]
            probability_map = probability_map[positive_idx]
            boxes_for_nms = np.rollaxis(np.stack([x, y, x + 12, y + 12]), 1, 0)
            boxes_for_nms = torch.tensor(boxes_for_nms).float().to(self.device)
            probability_map = torch.tensor(probability_map).float().to(self.device)
            boxes_after_nms_idx = nms(boxes_for_nms, probability_map, self.iou_th)
            b = boxes_for_nms[boxes_after_nms_idx].numpy().squeeze()
            scores.extend(probability_map[boxes_after_nms_idx].numpy())
            bounding_boxes.extend((b * self.scale ** i).astype(np.uint32))

        return bounding_boxes, scores

    def pyramid(self, image):
        """
        https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
        """
        # yield the original image
        yield image
        # keep looping over the pyramid
        while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / self.scale)
            image = imutils.resize(image, width=w)
            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < self.min_size[1] or image.shape[1] < self.min_size[0]:
                break
            # yield the next image in the pyramid
            yield image


model = NetFCN()
state_dict = torch.load('12Net_60.pt', map_location=torch.device('cpu'))['model_state_dict']

model.load_state_dict(state_dict)
detect = Detect(model, scale=1.1, min_size=(24, 24), iou_th=0.4)
# image = cv.imread('img.jpg')
# boxes, scores = detect.detect(image)
# overlay(image, boxes)
evaluate_model("data/fddb/FDDB-folds/FDDB-fold-01.txt", detect)
