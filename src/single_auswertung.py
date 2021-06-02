import time
from threading import Thread

import numpy as np
import cv2

from .darknet import Net
from .image_data import ImageData


class Spot:

    def __init__(self, camera_name, base_pos, drone_vec, size, single_time) -> None:
        self.camera_name = camera_name
        self.base_pos = base_pos
        self._drone_vec = drone_vec / np.linalg.norm(drone_vec)
        self.size = size
        self.single_time = single_time

    def __str__(self):
        return f"bPos: {self.base_pos}; dVec: {self.drone_vec}"

    @property
    def drone_vec(self):
        return self._drone_vec / np.linalg.norm(self._drone_vec)

    def copy(self, base_pos=None, drone_vec=None):
        if base_pos is None or drone_vec is None:
            base_pos = self.base_pos
            drone_vec = self.drone_vec
        return Spot(self.camera_name, base_pos, drone_vec, self.size, self.single_time)

    def get_2d(self, trans: np.array):
        base_pos = np.dot(trans, self.base_pos)
        drone_vec = np.dot(trans, self.drone_vec)
        return self.copy(base_pos=base_pos, drone_vec=drone_vec)

    def get_point(self, s):
        return self.base_pos + s * self.drone_vec

    @property
    def radius(self):
        return self.size / 2.0


class BildAuswertung:
    """Class to run YOLO on single images and extract possible drone spottings from them.
    """

    def __init__(self):
        """Initializes a BildAuswertung object.

        This usually includes initializing a Net (darknet) object with yolo data, which takes a while (~2s).
        """
        self.darknet = Net("/home/max/darknet_test/darknet/libdarknet.so",
                           "/home/max/darknet_test/darknet/yolov3.weights",
                           "/home/max/darknet_test/darknet/cfg/yolov3.cfg",
                           "/home/max/darknet_test/darknet/cfg/coco.data")

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
            spotted_list = self.darknet.detect(image_dict[key].image)
            # Iterate over found images      
            for raw_spot in spotted_list:
                if raw_spot[0] in [b"airplane", b"drone"] or True: # todo possibly extend todo remove teutology
                    height, width, _ = np.shape(image_dict[key].image)
                    base_pos = image_dict[key].pos
                    single_time = image_dict[key].timestamp
                    esp_name = image_dict[key].esp_name
                    spot_x, spot_y = (int(a)for a in raw_spot[2][:2])
                    # Transform pixel coordinates into 3d direction vector
                    # todo where exactly is top and bottom ?
                    drone_vec = np.array([(spot_x/width - 0.5) * 0.466, (spot_y/height - 0.5) * 0.466, 1])
                    size = np.average(np.array(raw_spot[2][2:])/np.array((height, width)))
                    spot = Spot(esp_name, base_pos, drone_vec, size, single_time)
                    spot_list.append(spot)
        return spot_list
        

    #def get_spotted(self):
    #    return self.get_spotted_dummy()

    def get_spotted_dummy(self):
        e1 = Spot(
            "cam 0",
            np.array([0, 0, 0]), # Base pos
            np.array([0, 0, 1]), # Drone vector
            1, # size
            #time.time() - 0.2 # single time
            1
        )
        e2 = Spot(
            "cam 1",
            np.array([1, 0, 0]), # Base pos
            np.array([0, 1, 1]), # Drone vector
            1, # size
            #time.time() - 0.2 # single time
            1
        )
        e3 = Spot(
            "cam 2",
            np.array([0, 1, 0]), # Base pos
            np.array([0, -0.7, 0.7]), # Drone vector
            1, # size
            #time.time() - 0.2 # single time
            1
        )
        return ([e1, e2], time.time())




def main():
    dummy_img = cv2.imread("/home/max/darknet_test/darknet/data/dog.jpg")
    dummy_img_data = ImageData(dummy_img, time.time(), "some_esp", np.array([0, 0, 0]))
    auswertung = BildAuswertung()
    print(auswertung.get_spotted({"some_esp": dummy_img_data}))

if __name__ == "__main__":
    main()
