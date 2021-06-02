import time

import cv2
import numpy as np

from src.image_data import ImageData
from src.single_auswertung import BildAuswertung


def main():
    dummy_img = cv2.imread("/home/max/darknet_test/darknet/data/eagle.jpg")
    dummy_img_data = ImageData(dummy_img, time.time(), "some_esp", np.array([0, 0, 0]))
    #print(np.shape(dummy_img_data.image))
    #cv2.imshow("dingledangle", dummy_img_data.image)
    #cv2.waitKey(1000)
    #return
    auswertung = BildAuswertung()
    spotted = auswertung.get_spotted({"some_esp": dummy_img_data})
    print(spotted[0])

if __name__ == "__main__":
    main()