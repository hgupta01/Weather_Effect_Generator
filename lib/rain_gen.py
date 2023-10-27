import cv2
import scipy
import random
import numpy as np
from pathlib import Path
from lib.gen_utils import (generate_noisy_image, 
                           centreCrop,
                           binarizeImage, 
                           bwAreaFilter,
                           apply_motion_blur,
                           zoom_image_and_crop, 
                           get_otsu_threshold, 
                           color_level_adjustment)


class RainGenUsingNoise:
    def genRainLayer(self,
                      h, 
                      w, 
                      noise_scale=0.5, 
                      noise_amount=0.25, 
                      zoom_layer=2.0,
                      blur_kernel_size=15, 
                      blur_angle=-60
                      ):
        layer = generate_noisy_image(h, w, sigma=noise_scale, p=noise_amount)

        if blur_kernel_size>0:
            layer = apply_motion_blur(layer.copy(), blur_kernel_size, int(blur_angle))

        if zoom_layer>1:
            layer = zoom_image_and_crop(layer.copy(), r = zoom_layer)
    
        th = get_otsu_threshold(layer.copy())
        layer = color_level_adjustment(layer.copy(), inBlack=th, inWhite=th+100, outWhite=250, inGamma=1.0)
        return layer
    

class RainGenUsingMasks:
    def __init__(self, mask_folder:str, ext="png"):
        self._mask_path_list = sorted(Path(mask_folder).glob("*."+ext))

    def genSingleLayer(self, scale=4, area = (10,500), blur=False, rotate=0):
        streak_file = random.choice(self._mask_path_list)
        streak = cv2.cvtColor(cv2.imread(str(streak_file)), cv2.COLOR_BGR2GRAY)
        hs, ws = streak.shape
        if scale>1:
            streak = cv2.resize(streak, (int(ws*scale), int(hs*scale)))
        
        if rotate!=0:
            M = cv2.getRotationMatrix2D((int(ws*scale)/2,int(hs*scale)/2), rotate, 1)
            streak = cv2.warpAffine(streak,M,(int(ws*scale), int(hs*scale))) 
            
        binarized_streak = binarizeImage(streak)
        mask = bwAreaFilter(binarized_streak, area_range=area)
        
        # radius=2*ceil(2*sigma)+1
        streak_masked = streak*mask
        if blur:
            streak_masked = scipy.ndimage.gaussian_filter(streak_masked, sigma=1, mode='reflect', radius=5)
        return streak_masked
    
    def genStreaks(self, reqH=720, reqW=1280, rotate=0, num_itr=10, scale=2, 
                   area=(50, 150), blur=False, resize=False, 
                   inGamma=1.0):
        layer = np.zeros((reqH, reqW))
        
        blur_kernel_size = 3
        blur_angle = np.random.randint(-60, 60)
        
        for i in range(num_itr):
            streak = self.genSingleLayer(scale=scale, area=area, rotate=rotate)
            if blur:
                streak = apply_motion_blur(streak.astype(float), blur_kernel_size, blur_angle)
            if resize:
                streak = cv2.resize(streak.astype(float), (reqW, reqH))
            streak = centreCrop(streak, reqH, reqW)
            tr = random.random() * 0.2 + 0.25
            layer = layer + streak*tr
            
        layer = color_level_adjustment(layer.copy(), inBlack=10, inWhite=100, inGamma=inGamma, outBlack=0, outWhite=200)
        return layer
    
    def genRainEffect(self, intensnity):
        rotate = random.randint(-30,30)
        if intensnity=='high':
            layer_far = self.genStreaks(reqH=720, reqW=1280, rotate=rotate, 
                                        num_itr=random.randint(40,75), scale=1, 
                                        area=(5, 150), blur=False, resize=False, inGamma=1.0)
            layer_close = self.genStreaks(reqH=720, reqW=1280, rotate=rotate, 
                                          num_itr=random.randint(15,30), scale=1, 
                                          area=(150, 450), blur=False, resize=False, inGamma=1.0)
            
        if intensnity=='mod':
            layer_far = self.genStreaks(reqH=720, reqW=1280, rotate=rotate, 
                                        num_itr=random.randint(15,25), scale=1, 
                                        area=(75, 150), blur=False, resize=False, inGamma=2.0)
            layer_close = self.genStreaks(reqH=720, reqW=1280, rotate=rotate, 
                                          num_itr=random.randint(4,10), scale=1, 
                                          area=(150, 500), blur=False, resize=False, inGamma=2.0)
        tr = random.random() * 0.2 + 0.25
        layer = layer_far + layer_close*tr
        return layer