import torch
import cv2 as cv
import numpy as np


def get_sample_from_picture(img):
    img = np.rollaxis(img, 2, 0)  # convert to channels_first
    ind_x = np.random.choice(range(img.shape[1] - 12))
    ind_y = np.random.choice(range(img.shape[2] - 12))
    return img[:, ind_x:ind_x + 12, ind_y:ind_y + 12]


def get_list_of_pictures_without_person(path):
    f = open(path, 'r')
    res = list()
    for line in f:
        line = line[:-1]
        if line[7] == '-':
            res.append(line[:6])
    f.close()
    return res


list_of_pictures = get_list_of_pictures_without_person("data/PASCL/picturesList.txt")
res = np.zeros((30000, 3, 12, 12))
for i in range(30000):
    ind = np.random.choice(range(len(list_of_pictures)))
    pic_name = list_of_pictures[ind]
    pic_path = "data/JPEGImages/" + pic_name + '.jpg'
    image = cv.imread(pic_path)
    sample = get_sample_from_picture(image)
    res[i] = sample
    if i % 100 == 0:
        print("Iter = {0}".format(i))

torch.save(torch.tensor(res), "data/patches/12negative.pt")
