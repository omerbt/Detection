import torch
from net12FCN import NetFCN
import cv2 as cv
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NetFCN()
state_dict = torch.load('12FCN_300.pt', map_location=torch.device('cpu'))['model_state_dict']
model.load_state_dict(state_dict)


def get_sample_from_picture(image, model):
    image = np.rollaxis(image, 2, 0)  # convert to channels_first
    image_to_model = torch.tensor(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_to_model.float()).squeeze()
    probability_map = torch.softmax(output, dim=1).numpy()[0]
    x, y = np.indices(output.shape[1:])
    positive_idx = probability_map > 0.5
    x, y = 2 * x[positive_idx], 2 * y[positive_idx]
    index = np.random.choice(len(x))
    x, y = x[index], y[index]
    return image[:, x:x+12, y:y+12]


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
res = np.zeros((30000, 3, 24, 24))
for i in range(30000):
    ind = np.random.choice(range(len(list_of_pictures)))
    pic_name = list_of_pictures[ind]
    pic_path = "data/JPEGImages/" + pic_name + '.jpg'
    image = cv.imread(pic_path)
    sample = get_sample_from_picture(image, model)
    sample = np.rollaxis(sample, 0, 3)
    sample = cv.resize(sample, dsize=(24, 24))
    sample = np.rollaxis(sample, 2, 0)
    res[i] = sample

torch.save(torch.tensor(res), "data/patches/24negative.pt")
