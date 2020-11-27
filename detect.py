import torch
from torchvision.ops import nms

from net12FCN import NetFCN
from noyNet import Net
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import imutils


class Detect:
    def __init__(self, model, scale, min_size, iou_th=0.1):
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
        for i, scaled_image in enumerate(self.pyramid(image)):
            scaled_image = np.rollaxis(scaled_image, 2, 0)  # convert to channels_first
            scaled_image = torch.tensor(scaled_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(scaled_image.float()).squeeze()
            z = output.view(-1, 2)
            print(z[0])
            probability_map = torch.softmax(z, dim=1).numpy()
            print(probability_map[0])
            return
            print(np.percentile(probability_map, 50))
            x, y = np.indices(output.shape[1:])
            x, y = x.flatten()[probability_map > 0.5], y.flatten()[probability_map > 0.5]
            x *= 4
            y *= 4
            probability_map = probability_map[probability_map > 0.5]
            boxes_for_nms = np.rollaxis(np.stack([x, y, x + 12, y + 12]), 1, 0)
            boxes_for_nms = torch.tensor(boxes_for_nms).float().to(self.device)
            probability_map = torch.tensor(probability_map).float().to(self.device)
            boxes_after_nms_idx = nms(boxes_for_nms, probability_map, self.iou_th)
            b = boxes_for_nms[boxes_after_nms_idx].numpy().squeeze()
            b *= self.scale ** i
            if i == 3:
                for x1, y1, x2, y2 in b:
                    cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plt.imshow(image)
                plt.show()
                return

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


# model = NetFCN()
model = Net()
# state_dict = torch.load('12NetFCN.pt', map_location=torch.device('cpu'))['model_state_dict']
state_dict = torch.load('n12_fcn_model_200_epoch.pt', map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)
model.load_state_dict(state_dict)
image = cv.imread('img.jpg')
detect = Detect(model, scale=1.5, min_size=(32, 32))
detect.detect(image)
