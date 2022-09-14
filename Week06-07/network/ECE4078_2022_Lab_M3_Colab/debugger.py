import sys, os
import os
import numpy as np
from PIL import Image
from detector import Detector
import matplotlib.pyplot as plt
import matplotlib.patches as label_box
from tqdm import tqdm
import os

test_dir = './test_lab_output/'
ckpt = '../scripts/model/model.best2.pth'

detector = Detector(ckpt, use_gpu=False)

pred_dir = os.path.join(test_dir, "pred")
os.makedirs(pred_dir, exist_ok=True)
all_test_images = [file for file in os.listdir(test_dir) if file.endswith('.png')]
for image_name in all_test_images:
    np_img = np.array(Image.open(os.path.join(test_dir, image_name)))
    pred, colour_map = detector.detect_single_image(np_img)
    title = ["Input", "Prediction"]
    pics = [np_img, colour_map]
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(pics[0], interpolation='nearest')
    axs[0].set_title(title[0])
    axs[1].imshow(pics[1], interpolation='nearest')
    axs[1].set_title(title[1])
    axs[0].axis('off')
    axs[1].axis('off')
    path = os.path.join(pred_dir, image_name)
    plt.show()
    # plt.savefig(os.path.join(pred_dir, image_name[:-4]+'.jpg'))
