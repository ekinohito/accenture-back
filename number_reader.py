import pytesseract
import cv2


pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


class Detector:
    def __init__(self, config):
        self.config = config

    def get_number(self, img):
        img = cv2.resize(img, (800, 600))
        number = pytesseract.image_to_string(img, config=self.config)

        return number

    def save_image(self, img):
        img = cv2.resize(img, (640, 640))

        boxes = pytesseract.image_to_data(img, config=self.config)

        for x, b in enumerate(boxes.splitlines()):
            if x!=0:
                b = b.split()
                if len(b) == 12:
                    x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                    cv2.rectangle(img, (x, y), (w + x, h + y), (0,0,255), 3)
                    cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

        return img
        cv2.imwrite('number_detector/res/' + image_path.split('/')[2], img)


    def save_video(self, video_path):
        cap = cv2.VideoCapture(video_path)  

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w, h = 300, 200

        print("Total frames: {0}, Frames per second: {1}".format(frame_count, fps))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('number_detector/res/' + self.image_path.split('/')[1], fourcc, fps, (w, h))

        count = 0
        while True:
            success, img = cap.read()
            if success:
                img = cv2.resize(img, (300, 200))
                boxes = pytesseract.image_to_data(img, config=self.config)

                for x, b in enumerate(boxes.splitlines()):
                    if x!=0:
                        b = b.split()
                        if len(b) == 12:
                            x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                            cv2.rectangle(img, (x, y), (w + x, h + y), (0,0,255), 3)
                            cv2.putText(img, b[11], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 2)

                writer.write(img)
                count += 1
                print(count)
            else:
                break

        cap.release()
        writer.release() 


if __name__ == '__main__':
    detector = Detector(r'--oem 3 --psm 11 outputbase digits')
    image_path = './num1.jpg'
    img = cv2.imread(image_path)
    res = detector.get_number(img)
    print(res)

    # detector.save_image('number_detector/numbers/num1.jpg')