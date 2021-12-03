#!bin/python

from asyncio.tasks import create_task
import cv2
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from number_reader import Detector
from itertools import cycle
from asyncio import sleep

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
number_detector = Detector(r'--oem 3 --psm 11 outputbase digits')
side_images = [cv2.imread('photos/num1.jpg'), cv2.imread('photos/num2.jpg'), cv2.imread('photos/num3.jpg')]
current_side_image = None

@app.get("/")
async def root():
    return {"message": "Hello World"}

counter = 0
@app.get("/number")
async def get_number():
    global counter
    counter += 1
    cv2.imwrite(f'public/{counter}.jpg', number_detector.save_image(current_side_image))
    return {"url": f'public/{counter}.jpg', "code": number_detector.get_number(current_side_image)}


async def iterate():
    global current_side_image
    for img in cycle(side_images):
        current_side_image = img
        await sleep(1)

create_task(iterate())