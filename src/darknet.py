from ctypes import *
import math
import random
import time
import os

import cv2

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Net:
    
    #lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)

    def _setup(self, darknet_so_path: str):
        self.lib = CDLL(darknet_so_path, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    def _detect(self, net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(net, im)
        dets = self.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, meta.classes, nms);

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def get_net(self, config_path: str, weights_path: str):
        return self.load_net(c_char_p(config_path.encode("utf-8")), c_char_p(weights_path.encode("utf-8")), 0)

    def get_meta(self, meta_path: str):
        return self.load_meta(c_char_p(meta_path.encode("utf-8")))

    def get_detections(self, net, meta, file: str) -> list:
        return self._detect(net, meta, c_char_p(file.encode("utf-8")))

    def __init__(
        self,
        darknet_so_path: str,
        weights_path: str,
        config_path: str,
        meta_path: str
    ) -> None:
        self._setup(darknet_so_path)
        self._net = self.get_net(config_path, weights_path)
        self._meta = self.get_meta(meta_path)

    def make_detection(self, picture_path: str):
        return self.get_detections(self._net, self._meta, picture_path)

    def detect(self, img) -> list:
        cv2.imwrite("temp_pic.png", img)
        detection = self.make_detection("temp_pic.png")
        os.remove("temp_pic.png")
        return detection

    
if __name__ == "__main__":
    #net = get_net("./cfg/yolov3.cfg", "./yolov3.weights")
    #meta = get_meta("./cfg/coco.data")
    #r = get_detections(net, meta, "data/dog.jpg")
    #print(r)
    # /home/max/darknet_test/darknet/libdarknet.so
    darknet = Net("/home/max/darknet_test/darknet/libdarknet.so",
                  "/home/max/darknet_test/darknet/yolov3.weights",
                  "/home/max/darknet_test/darknet/cfg/yolov3.cfg",
                  "/home/max/darknet_test/darknet/cfg/coco.data")
    print(darknet.make_detection("/home/max/darknet_test/darknet/data/dog.jpg"))
    img = cv2.imread("/home/max/darknet_test/darknet/data/eagle.jpg")
    start_time = time.time()
    #print(darknet.make_detection("/home/max/darknet_test/darknet/data/eagle.jpg"))
    print(darknet.detect(img))
    print(time.time()-start_time)
