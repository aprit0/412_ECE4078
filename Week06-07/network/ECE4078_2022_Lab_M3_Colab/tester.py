import numpy as np
from PIL import Image
from detector import Detector
import os
from tqdm import tqdm


test_dir = './test_lab_output/'
ckpt = '../scripts/model/model.best2.pth'

detector = Detector(ckpt, use_gpu=False)

all_test_images = [file for file in os.listdir(test_dir) if file.endswith('.png')]
for image_name in all_test_images:
    np_img = np.array(Image.open(os.path.join(test_dir, image_name)))
    print(np_img.shape)
    pred, colour_map = detector.detect_single_image(np_img)
    print(pred.shape, type(pred))
    im = Image.fromarray(pred.astype('uint8'))
    im.save("out/{}".format(image_name))
