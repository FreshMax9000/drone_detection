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
        if base_pos is None or drone_vec is None:
            base_pos = self.base_pos
            drone_vec = self.drone_vec
        return Spot(self.camera_name, base_pos, drone_vec, self.size, self.single_time)

    def get_2d(self, trans: np.ndarray):
        #base_pos = np.dot(trans, self.base_pos)
        #drone_vec = np.dot(trans, self.drone_vec)
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

    def __init__(self, conf: float, use_scissors=False) -> None:
        """Initializes a BildAuswertung object.

        This usually includes initializing a Net (darknet) object with yolo data, which takes a while (~2s).
        """
        if use_scissors:
        #self.yolov5_model = attempt_load(weight_path)
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path="/home/max/Documents/drone_detection/yolov5m.pt")
        else:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", path="/home/max/Documents/drone_detection/yolov5m_drones.pt")
        self.model.conf = conf
        self.device = select_device('0')
        self.scissors = use_scissors
        #self.model = attempt_load(weight_path, map_location=self.device, inplace=True)
        #self.model.eval()
        #self.stride = int(self.model.stride.max())
        #imgsz = 640
        #if self.device.type != 'cpu':
        #    self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))

        #self.darknet = Net("/home/max/darknet_test/darknet/libdarknet.so",
        #                   "/home/max/darknet_test/darknet/yolov3.weights",
        #                   "/home/max/darknet_test/darknet/cfg/yolov3.cfg",
        #                   "/home/max/darknet_test/darknet/cfg/coco.data")

    def detect_yolov5(self, image):
        #image = np.moveaxis(image, (0, 1, 2), (1, 2, 0))
        #image = np.array([image])
        
        #img = torch.from_numpy(image).to(self.device)
        #img = img.float()
        #img /= 255.0
        #if img.ndimension() == 3:
        #    img = img.unsqueeze(0)
        results = self.model(image)
        return results.xyxy[0].cpu().numpy()[:,:]

    @staticmethod
    def draw_bounding_box(image, x1, y1, x2, y2, c):
        #print(f"x: {x}, y: {y}, w: {w}, h: {h}")
        #w_half = int(w / 2)
        #h_half = int(h / 2)
        #cv2.rectangle(image,(x-w_half,y-h_half),(x+w_half,y+h_half),(255,0,0),2)
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
            # Iterate over found images      
            for raw_spot in spotted_list:
                #if raw_spot[5] in [0, 4]: # drone/person, airplane and not scissors (76)
                if self.scissors and raw_spot[5] in [4, 76] or not self.scissors and raw_spot[5] in [0, 4]:
                    height, width, _ = np.shape(image_dict[key].image)
                    base_pos = image_dict[key].pos
                    single_time = image_dict[key].timestamp
                    esp_name = image_dict[key].esp_name
                    x1, y1 = (int(a)for a in raw_spot[:2])
                    x2 = int(raw_spot[2])
                    y2 = int(raw_spot[3])
                    # draw bounding box
                    self.draw_bounding_box(image_dict[key].image, x1, y1, x2, y2, raw_spot[4])
                    # Transform pixel coordinates into 3d direction vector
                    # todo where exactly is top and bottom ?
                    geka_x = 0.5 # Gegenkathete
                    geka_y = 0.38
                    drone_vec = np.array([((x1+x2)/(2*width) - 0.5) * geka_x, ((y1+y2)/(2*height) - 0.5) * -geka_y, 1])
                    size = np.average(np.array(raw_spot[2:4])/np.array((height, width)))
                    spot = Spot(esp_name, base_pos, drone_vec, size, single_time)
                    spot_list.append(spot)
        return spot_list, image_dict
        

    #def get_spotted(self):
    #    return self.get_spotted_dummy()






def main():
    dummy_img = cv2.imread("/home/max/darknet_test/darknet/data/dog.jpg")
    dummy_img_data = ImageData(dummy_img, time.time(), "some_esp", np.array([0, 0, 0]))
    auswertung = BildAuswertung()
    print(auswertung.get_spotted({"some_esp": dummy_img_data}))

if __name__ == "__main__":
    main()
