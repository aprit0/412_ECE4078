import numpy as np
from PIL import Image
from detector import Detector
import os
import torch
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.style.use( 'ggplot' )
from tqdm import tqdm
import sys
sys.path.insert(0, 'yolov5')

test_dir = '../ECE4078_2022_Lab_M3_Colab/test_lab_output/'
ckpt = os.getcwd() + '/model/best_simWorld.pt'


all_test_images = [file for file in os.listdir(test_dir) if file.endswith('.png')]
model = torch.hub.load('./yolov5', 'custom', path=ckpt, source='local')
for image_name in all_test_images:
    np_img = np.array(Image.open(os.path.join(test_dir, image_name)))
    out = model(np_img)
    print(image_name, out.print(), out.xyxy)
    fig, ax = plt.subplots()
    ax.imshow(np_img)
    pose = out.xyxy[0]
    for i in pose:
        min = (i[0], i[1])
        width = i[2] - i[0]
        height = i[3] - i[1]
        print(min, i)
        rect = patches.Rectangle(min, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig('im/{}'.format(image_name))
