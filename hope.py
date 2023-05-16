from PIL import Image, ImageOps
import os
import numpy as np
import cv2


IMAGE_SIZE = 321

args = update_parser(parser)

def preprocess(img):
    keep_ratio = args.keep_ratio

    im_h, im_w, _ = img.shape

    if keep_ratio:
        scale = IMAGE_SIZE / max(im_h, im_w)
        ow, oh = int(im_w * scale), int(im_h * scale)
        if ow != im_w or oh != im_h:
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

        if ow != IMAGE_SIZE or oh != IMAGE_SIZE:
            pad = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
            pad_h, pad_w = pad.shape[:2]
            pad_h = (pad_h - oh) // 2
            pad_w = (pad_w - ow) // 2
            pad[pad_h:pad_h + oh, pad_w:pad_w + ow] = img
            img = pad
    else:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    img = img / 255
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img


def mirror(path_to_origin, img_name, output_folder):
    im = Image.open(path_to_origin + img_name)
    ImageOps.mirror(im).save(output_folder+"mirror"+img_name, "JPEG")

def rotate(path_to_origin, img_name, output_folder):
    im = Image.open(path_to_origin + img_name)
    im.rotate(-30).save(output_folder+"rotate_"+img_name, "JPEG")

path_to_origin = "sources/"
output_folder = "Augmentation_result/"
img_names = os.listdir(path_to_origin)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for img in img_names:
    rotate(path_to_origin, img, output_folder)
    mirror(path_to_origin, img, output_folder)









'''
WEIGHT_PATH = 'landmarks_classifier_asia_V1_1.onnx'
MODEL_PATH = 'landmarks_classifier_asia_V1_1.onnx.prototxt'
'''