#!bin/python

from asyncio.tasks import create_task
import cv2
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from number_reader import Detector
from itertools import cycle
from asyncio import sleep

from rubbish_classifier.inference import predictions

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
number_detector = Detector(r'--oem 3 --psm 11 outputbase digits')
side_images = [cv2.imread('photos/num1.jpg'), cv2.imread('photos/num2.jpg'), cv2.imread('photos/num3.jpg')]
top_images = [[cv2.imread('photos/test1.jpg')], [cv2.imread('photos/test2.jpg')], [cv2.imread('photos/test3.jpg')]]
current_side_image = None
current_top_image = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

counter = 0
@app.get("/number")
async def get_number():
    global counter
    counter += 1
    counter %= 1000
    cv2.imwrite(f'public/{counter}.jpg', number_detector.save_image(current_side_image))
    return {"url": f'public/{counter}.jpg', "code": number_detector.get_number(current_side_image)}


@app.get("/frame")
async def get_frame():
    global counter
    counter += 1
    counter %= 1000
    probability = predictions(current_top_image)
    cv2.imwrite(f'public/{counter}.jpg', current_top_image)
    return {"url": f'public/{counter}.jpg', "probability": probability}



async def iterate():
    global current_side_image
    global current_top_image
    for side_img, top_imgs in cycle(zip(side_images, top_images)):
        current_side_image = side_img
        for top_img in top_imgs:
            current_top_image = top_img
            await sleep(3)

create_task(iterate())