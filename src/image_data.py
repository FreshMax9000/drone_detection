
import cv2
import numpy as np


class ImageData:

    folder_path = "/home/max/Documents/Studienarbeit/DroneSpot/implementation_kek/a_lot_of_images"

    def __init__(self, image, timestamp: float, esp_name: str, pos: np.array) -> None:
        self.image = image
        self.timestamp = timestamp
        self.esp_name = esp_name
        self.pos = pos

    def save_to_file(self):
        cv2.imwrite(self.folder_path+f"/{self.timestamp}-{self.esp_name}.jpg", self.image)
