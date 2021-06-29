"""This module represents the detection of drones in single images.

It therefore represents the middle step in the detection pipeline. The class Spot
represents one detection of one drone in one image and the class BildAuswertung
does the whole detection procedure."""
import time
from threading import Thread

import numpy as np
import cv2
import torch

from .darknet import Net
from .image_data import ImageData
from yolov5.models.experimental import attempt_load
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import check_img_size


class Spot:
    """Class representing one drone detection in one image.
    
    The detection is represented by the position of the microcontroller (base_pos) and the direction
    the drone was spotted from microcontroller (drone_vec). This class also contains the name of
    the microcontroller which spotted the drone, the size of the drone (0=no drone; 1=drone is filling
    the whole screen) and the approximate time the single detection was made (single_time).
    """

    def __init__(self, camera_name, base_pos, drone_vec, size, single_time) -> None:
        self.camera_name = camera_name
        self.base_pos = base_pos
        self._drone_vec = drone_vec
        self.size = size
        self.single_time = single_time

    def __str__(self):
        return f"[{self.camera_name}] bPos: {self.base_pos}; dVec: {self.drone_vec}"

    @property
    def drone_vec(self):
        return self._drone_vec / np.linalg.norm(self._drone_vec)

    @property
    def raw_drone_vec(self):
        return self._drone_vec

    def copy(self, base_pos=None, drone_vec=None):
        """Returns a copy of itself.
        
        If a new base_pos and drone_vec is given, the copy has these values instead of
        the values from itself.
        """
        if base_pos is None or drone_vec is None:
            base_pos = self.base_pos
            drone_vec = self.drone_vec
        return Spot(self.camera_name, base_pos, drone_vec, self.size, self.single_time)

    def get_2d(self, trans: np.ndarray):
        """Returns a 2d version of itself based on the given transformation matrix.
        """
        base_pos = np.dot(trans, self.base_pos)
        drone_vec = np.dot(trans, self.drone_vec)
        return self.copy(base_pos=base_pos, drone_vec=drone_vec)

    def get_point(self, s):
        """Returns a point on itself.
        
        In this case, a Spot is seen as a straight line in space. base_vec is the support vector
        and drone_vec is the direction vector. The return value of this function is equal to:
        return = base_vec + s * drone_vec
        """
        return self.base_pos + s * self.drone_vec

    @property
    def radius(self):
        return self.size / 2.0


class BildAuswertung:
    """Class to run YOLO on single images and extract possible drone spottings from them.
    """

    def __init__(self, conf: float, use_scissors=False) -> None:
        """Initializes a BildAuswertung object.

        This usually includes initializing a model from pytorch with yolo data, which takes a while (~2s).
        conf (confidence) is the threshhold value required for detection. If conf is 0.3, the system will only
        use detection where its more or exactly 30% sure that it detected a drone.
        use_scissors tries to detect scissors instead of drones, which is useful for debugging.
        """
        if use_scissors:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path="/home/max/Documents/drone_detection/yolov5m.pt")
        else:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path="/home/max/Documents/drone_detection/yolov5m_drones.pt")
        self.model.conf = conf
        self.device = select_device('0')
        self.scissors = use_scissors

    def detect_yolov5(self, image):
        """Use the yolov5 model to make detections."""
        results = self.model(image)
        return results.xyxy[0].cpu().numpy()[:,:]

    @staticmethod
    def draw_bounding_box(image, x1, y1, x2, y2, c):
        """Draws a bounding box with the diagonal corners (x1, y1) and (x2, y2) and the confidence c."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(c), (x1, y2), font, 1, (255, 0, 0), thickness=2)
        cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),2)

    def get_spotted(self, image_dict: dict) -> list:
        """Generates a list of Spot objects from the given image_dict.

        All Images in image_dict are searched for drone objects. Each found drone object is
        described by a Spot object. A list containing all found Spots is then returned

        Args:
            image_dict (dict): A dict containing ImageData objects; keys are the names of
                the corresponding ESPs

        Returns:
            list: A list containing Spot objects, which correspond the found drones in
                the given images.
        """
        spot_list = []
        # Iterate over all images
        for key in image_dict:
            spotted_list = self.detect_yolov5(image_dict[key].image)
            # Iterate over detection made in images   
            for raw_spot in spotted_list:
                if self.scissors and raw_spot[5] in [4, 76] or not self.scissors and raw_spot[5] in [0, 4]:
                    # Transform the raw spot (pixelcoordinates on an image) into a more usable
                    # base_pos + drone_vec form
                    height, width, _ = np.shape(image_dict[key].image)
                    base_pos = image_dict[key].pos # base_pos is simply position of microcontroller
                    single_time = image_dict[key].timestamp
                    esp_name = image_dict[key].esp_name
                    x1, y1 = (int(a)for a in raw_spot[:2])
                    x2 = int(raw_spot[2])
                    y2 = int(raw_spot[3])
                    # draw bounding box
                    self.draw_bounding_box(image_dict[key].image, x1, y1, x2, y2, raw_spot[4])
                    # Transform pixel coordinates into 3d direction vector
                    geka_x = 0.5 # Gegenkathete x
                    geka_y = 0.38 # Gegenkathete y
                    # Calculating a "ray" which gets shot through the screen of the camera to the spotted object
                    drone_vec = np.array([((x1+x2)/(2*width) - 0.5) * geka_x * 2, ((y1+y2)/(2*height) - 0.5) * -geka_y * 2, 1])
                    size = np.average(np.array(raw_spot[2:4])/np.array((height, width)))
                    spot = Spot(esp_name, base_pos, drone_vec, size, single_time)
                    spot_list.append(spot)
        return spot_list, image_dict
        

def main():
    dummy_img = cv2.imread("/home/max/darknet_test/darknet/data/dog.jpg")
    dummy_img_data = ImageData(dummy_img, time.time(), "some_esp", np.array([0, 0, 0]))
    auswertung = BildAuswertung()
    print(auswertung.get_spotted({"some_esp": dummy_img_data}))

if __name__ == "__main__":
    main()
