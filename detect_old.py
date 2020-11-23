from net12FCN import NetFCN
import torch
from torchvision.ops import nms as nms
import imutils
import cv2 as cv
import numpy as np


class Detect:
    def __init__(self, model, state_dict, scale, min_size,
                 window_size, step_size, batch_size, iou_th):
        self.model = model
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.scale = scale
        self.min_size = min_size
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iou_th = iou_th

    def detect(self, image):
        bounding_boxes = []
        # loop over the image pyramid
        for i, scaled_image in enumerate(self.pyramid(image)):
            windows, lefts, rights = [], [], []
            g = self.sliding_window(scaled_image)
            for left, right, window in iter(g):  # self.sliding_window(scaled_image):
                lefts.append(left)
                rights.append(right)
                windows.append(np.rollaxis(window, 2, 0))

            lefts, rights, windows = np.array(lefts), np.array(rights), np.array(windows)

            batches = np.array_split(windows, int(len(windows) / self.batch_size) + 1)
            outputs = []
            for batch in batches:
                # print(batch.dtype)
                batch = torch.from_numpy(batch).float().to(self.device)
                with torch.no_grad():
                    outputs.append(self.model(batch))
            # model prediction for all windows in this scale

            outputs = torch.cat(outputs, dim=0)




            # TODO check if this is necessary
            print(outputs.shape)
            print("************")
            outputs = torch.softmax(outputs, dim=1).numpy().squeeze()
            print(outputs)
            # outputs = torch.sigmoid(outputs).numpy().squeeze()
            probs = outputs[:, 0]
            lefts, rights = lefts[probs > 0.5], rights[probs > 0.5]
            # print(-2)
            boxes_for_nms = torch.from_numpy(np.array([[x, y, x + self.window_size, y + self.window_size]
                                                       for x, y in zip(lefts,rights)])).float()
            # print(-1)
            probs = probs[probs > 0.5]
            # print(0)
            boxes_after_nms_idx = nms(boxes_for_nms, torch.tensor(probs).float(), self.iou_th)
            # print(1)
            b = boxes_for_nms[boxes_after_nms_idx].numpy().squeeze()
            # print(type(b))
            bounding_boxes.append(b * (self.scale ** i))
        bounding_boxes = np.array(bounding_boxes).squeeze()
        print(bounding_boxes.shape)
        cv.imshow("jhgjhgjk", image)
        # import time
        # time.sleep(5)
        for x1, y1, x2, y2 in bounding_boxes:
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv.imshow("jhgjhgjk", image)
        # cv.waitKey(1)
        # import time
        # time.sleep(100)
        import matplotlib.pyplot as plt
        plt.imshow(image)
        plt.show()

    def pyramid(self, image):
        """
        https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
        """
        # yield the original image
        yield image
        # keep looping over the pyramid
        # while True:
        #     # compute the new dimensions of the image and resize it
        #     w = int(image.shape[1] / self.scale)
        #     image = imutils.resize(image, width=w)
        #     # if the resized image does not meet the supplied minimum
        #     # size, then stop constructing the pyramid
        #     if image.shape[0] < self.min_size[1] or image.shape[1] < self.min_size[0]:
        #         break
        #     # yield the next image in the pyramid
        #     yield image

    def sliding_window(self, image):
        """
        based on https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
        """
        # slide a window across the image
        for x in range(0, image.shape[0], self.step_size):
            if x + self.window_size > image.shape[0]:
                return
            for y in range(0, image.shape[1], self.step_size):
                # yield the current window
                if y + self.window_size <= image.shape[1]:
                    yield x, y, image[x:x + self.window_size, y:y + self.window_size]


model = NetFCN()
state_dict = torch.load('12NetFCN.pt', map_location=torch.device('cpu'))['model_state_dict']
import matplotlib.pyplot as plt
image = cv.imread('img.jpg')

detect = Detect(model, state_dict, scale=1.2, min_size=250,
                window_size=12, step_size=4, batch_size=256, iou_th=0.5)
detect.detect(image)
# cv.imshow("jhgjhgjk", image)
# # import time
# # time.sleep(5)
# # for x1, y1, x2, y2 in bounding_boxes:
# #     cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
# # cv.imshow("jhgjhgjk", image)
# cv.waitKey(1)
# import time
#
# time.sleep(100)