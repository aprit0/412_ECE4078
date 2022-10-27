import os 
import time

import cmd_printer
import numpy as np
import torch
from args import args
from res18_skip import Resnet18Skip
from torchvision import transforms
import cv2
import random

class Detector:
    def __init__(self, ckpt, use_gpu=False):
        self.args = args
        #self.ckpt = ckpt
        #self.model = Resnet18Skip(args)
        self.model = torch.hub.load('./yolov5', 'custom', path=ckpt, source='local', force_reload=True)
        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False
        #self.load_weights(ckpt)
        #self.model = self.model.eval()
        cmd_printer.divider(text="warning")
        print('This detector uses "RGB" input convention by default')
        print('If you are using Opencv, the image is likely to be in "BRG"!!!')
        cmd_printer.divider()
        self.colour_code = np.array([(220, 220, 220), (128, 0, 0), (128, 128, 0), (0, 128, 0), (192, 68, 0), (0, 0, 255)])

    def detect_single_image(self, np_img):
        torch_img = self.np_img2torch(np_img)
        tick = time.time()
        with torch.no_grad():
            pred = self.model(np_img)
            # if self.use_gpu:
            #     pred = torch.argmax(pred.squeeze(),
            #                         dim=0).detach().cpu().numpy()
            # else:
            #     pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()

        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")
        detPandas = pred.pandas().xyxy[0]
        # yolo colormap

        if not detPandas.empty:
            # deleting rows if conf <0.75
            print('Pandas LOW CONFIDENCES: ', detPandas[detPandas.confidence<0.75])
            detPandas = detPandas.drop(detPandas[detPandas.confidence<0.85].index)
            try:
                detPandas = detPandas.drop(detPandas.index[1:])
            except:
                pass
            print("\ndetPandas", detPandas)
            colour_map = self.visualise_yolo(np_img, detPandas)
        else:
            colour_map = cv2.resize(np_img, (320, 240), cv2.INTER_NEAREST)
        #print(np.shape(color_mapYolo))
        #colour_map = self.visualise_output(pred)

        return detPandas, colour_map

    def visualise_yolo(self, np_img, nn_output):
        # get bounding boxes
        colours = []
        for idx in nn_output.index:
            color = [random.randint(0, 255) for i in range(3)]
            xA = int(nn_output.xmin[idx])
            xB = int(nn_output.xmax[idx])
            yA = int(nn_output.ymin[idx])
            yB = int(nn_output.ymax[idx])
            # drawing bounding box on the image
            predImg = cv2.rectangle(np_img, (xA,yA), (xB,yB), color, 2)
            colours.append(color)
        # resizing the image
        color_map = cv2.resize(predImg, (320, 240), cv2.INTER_NEAREST)
        f_names = list(nn_output.name)
        for i in range(len(colours)):
           cv2.putText(color_map, str(f_names[i]), (10, 30 * i + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colours[i], 1)
        # wring text on bounding box
        return color_map



    def visualise_output(self, nn_output):
        r = np.zeros_like(nn_output).astype(np.uint8)
        g = np.zeros_like(nn_output).astype(np.uint8)
        b = np.zeros_like(nn_output).astype(np.uint8)
        for class_idx in range(0, self.args.n_classes + 1):
            idx = nn_output == class_idx
            r[idx] = self.colour_code[class_idx, 0]
            g[idx] = self.colour_code[class_idx, 1]
            b[idx] = self.colour_code[class_idx, 2]
        colour_map = np.stack([r, g, b], axis=2)
        colour_map = cv2.resize(colour_map, (320, 240), cv2.INTER_NEAREST)
        w, h = 10, 10
        pt = (10, 160)
        pad = 5
        labels = ['apple', 'lemon', 'pear', 'orange', 'strawberry']
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for i in range(1, self.args.n_classes + 1):
            c = self.colour_code[i]
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h),
                            (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
            colour_map  = cv2.putText(colour_map, labels[i-1],
            (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (0, 0, 0))
            pt = (pt[0], pt[1]+h+pad)
        return colour_map

    def load_weights(self, ckpt_path):
        ckpt_exists = os.path.exists(ckpt_path)
        if ckpt_exists:
            ckpt = torch.load(ckpt_path,
                              map_location=lambda storage, loc: storage)
            #self.model.load_state_dict(ckpt['weights'])
        else:
            print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                        # transforms.ColorJitter(brightness=0.4, contrast=0.3,
                                        #                         saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(
                                             mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu:
            img = img.cuda()
        return img
