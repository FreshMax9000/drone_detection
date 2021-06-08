#!/usr/bin/python3

import threading
import time
from typing import Callable, Iterable, Mapping, Tuple, Dict
import copy

import cv2
import numpy as np
from skimage import io

from .image_data import ImageData


class GetImageThread(threading.Thread):

   image_dict: dict = {}
   image_dict_lock: threading.Lock = threading.Lock()
   exit = False

   def __init__(
      self,
      esp_url: str,
      esp_name: str,
      pos: np.array,
      group: None = None,
      target: Callable = None,
      name: str = None,
      args: Iterable = (),
      kwargs: Mapping = {},
      *,
      daemon: bool = None
      ) -> None:
      """Intialize GetImageThread object.

      Args:
          esp_url (str): URL to ESP, eg 'http://192.168.188.80/capture'
          esp_name (str): Name of the ESP this Thread is responsible for
          pos (np.array): 3d position of this threads esp
          group (None, optional): [description]. Defaults to None.
          target (Callable, optional): [description]. Defaults to None.
          name (str, optional): [description]. Defaults to None.
          args (Iterable, optional): [description]. Defaults to ().
          kwargs (Mapping, optional): [description]. Defaults to {}.
          daemon (bool, optional): [description]. Defaults to None.
      """
      if name is None:
         name = f"{esp_name}-Thread"
      super().__init__(group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
      self.esp_url = esp_url
      self.esp_name = esp_name
      self.pos = pos

   def _get_frame(self):
      img = io.imread(self.esp_url)
      img = cv2.flip(img, 0)
      return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   def run(self):
      while not self.exit:
         img = self._get_frame()
         img_data = ImageData(img, time.time(), self.esp_name, self.pos)
         self.image_dict_lock.acquire(blocking=True)
         #print(f"Lock acquired by: {self.name}")
         self.image_dict.update({self.esp_name: img_data})
         self.image_dict_lock.release()





class ThreadController:
   """Class to control threads getting pictures from the esp32s.
   """

   def __init__(self, esp32_url_list: list, esp32_pos_list: list, esp32_name_list: list = None):
      """Initialize ThreadController objekct

      Args:
          esp32_url_list (list): List of URLs to access the image data of the esp32s
          esp32_pos_list (list): List of positions of the esp32s. Preferably in the same
            order as esp32_url_list
          esp32_name_list (list, optional): List of names for the esp32s. Preferably in the same
            order as esp32_url_list. Defaults to None.
      """
      self.threads = []
      for i, url in enumerate(esp32_url_list):
         if esp32_name_list is None:
            self.threads.append(GetImageThread(url, f"ESP{i}", esp32_pos_list[i]))
         else:
            self.threads.append(GetImageThread(url, esp32_name_list[i], esp32_pos_list[i]))

   def start_threads(self):
      for thread in self.threads:
         thread.start()

   def get_image_dict(self) -> Tuple[Dict[str, ImageData], float]:
      """Retrievies the most up-to-date pictures of the esp32s and returns them in a dict with a timestamp.

      Returns:
          tuple:
            dict: A dict mapping the names of the esp32s to ImageData objects. Usually looks somewhat like this:
            {"esp1": image_data_obj_1,
            "esp": image_data_obj_2}
         float: Timestamp when the images have been acquired from the ESPs
      """
      GetImageThread.image_dict_lock.acquire(blocking=True)
      #print(f"Lock acquired by: ThreadController.get_image_dict()")
      glob_time = time.time()
      image_dict = copy.copy(GetImageThread.image_dict)
      GetImageThread.image_dict_lock.release()
      return image_dict, glob_time

   def stop_all_threads(self):
      GetImageThread.exit = True

   def __enter__(self):
      self.start_threads()
      return self

   def __exit__(self, exception_type, exception_value, exception_traceback):
      self.stop_all_threads()


if __name__ == "__main__":
   url_list = ["http://10.42.0.124/capture", "http://10.42.0.206/capture"]
   name_list = ["uff", "kek"]
   controller = ThreadController(url_list, name_list)
   with controller as c:     
      for i in range(5):
         images = c.get_image_dict()
         for key in images:
            images[key].save_to_file()
         time.sleep(1)
