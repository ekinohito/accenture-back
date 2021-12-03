import cv2
from skimage.morphology import closing, square
from skimage.measure import label, regionprops
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


class DetectRubbish:
    def __init__(self, image_width=320, image_heigh=480, threshold=200, area=300):
        self.image_width = image_width
        self.image_heigh = image_heigh
        self.threshold = threshold
        self.area = area

    def detect(self, img, output_path):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_width, self.image_heigh))
        image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image = 255 - image

        # apply threshold
        bw = closing(image > self.threshold, square(3))

        # label image regions
        label_image = label(bw)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(img)

        rubbish = 0
        for i, region in enumerate(regionprops(label_image)):
            # take regions with large enough areas
            if region.area >= self.area:
                #print(region.bbox)

                temp_rubbish = (region.bbox[3]-region.bbox[1]) * (region.bbox[2]-region.bbox[0])
                rubbish += temp_rubbish

                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)


        rubbish_percent = round(rubbish / (self.image_width * self.image_heigh) * 100, 2)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(output_path)

        return rubbish_percent


if __name__ == '__main__':

    detector = DetectRubbish()
    res = detector.detect(input_path='photos/test3.jpg', 
                          output_path='public/res.jpg')

    print("Брака на фото: {0} %".format(res))