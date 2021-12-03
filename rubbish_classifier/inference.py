import cv2
import torch
from .model import model
from .data_transform import transform_test


def predictions(image, label=False):
    # image = Image.open(image)
    # image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform_test(image=image)['image']

    output = model(image[None, ...])

    if label:
      predicted = torch.round(output)
      predicted = predicted.squeeze().tolist()

      return predicted

    predicted_proba = round(output.squeeze().tolist() * 100, 2)
    return predicted_proba


if __name__ == '__main__':
    res = predictions(cv2.imread('photos/test1.jpg'))
    print("Вероятность брака: {0} %".format(res))