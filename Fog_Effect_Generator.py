import os
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from skimage import color
from tqdm.auto import tqdm

from lib.lime import LIME
from lib.fog_gen import fogAttenuation

from lib.gen_utils import (
                           illumination2opacity, 
                           reduce_lightHSV, 
                           scale_depth)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--depth_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--save_folder", type=str, default="./generated/", help="path to the folder")
    parser.add_argument("--txt_file", default=None, help="path to the folder")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()



class FogEffectGenerator:
    def __init__(self):
        self._lime = LIME(iterations=25, alpha=1.0)
        # self._illumination2darkness = {0: 1, 1: 0.75, 2: 0.65, 3:0.5}
        self._illumination2darkness = {0: 1, 1: 0.9, 2: 0.8, 3: 0.7}
        self._weather2visibility = (500, 2000)
        # self._illumination2fogcolor = {0: (80, 120), 1: (120, 160), 2: (160, 200), 3: (200, 240)}
        self._illumination2fogcolor = {0: (150, 180), 1: (180, 200), 2: (200, 240), 3: (200, 240)}
    
    def getIlluminationMap(self, img: np.ndarray) -> np.ndarray: 
        self._lime.load(img)
        T = self._lime.illumMap()
        return T

    def getIlluminationMapCheat(self, img: np.ndarray) -> np.ndarray: 
        T = color.rgb2gray(img)
        return T
    
    def genEffect(self, img_path: str, depth_path: str):
        I = np.array(Image.open(img_path))
        D = np.load(depth_path)

        hI, wI, _ = I.shape
        hD, wD = D.shape
        
        if hI!=hD or wI!=wD:
            D = scale_depth(D, hI, wI)
        
        # T = self.getIlluminationMap(I)
        T = self.getIlluminationMapCheat(I)
        illumination_array = np.histogram(T, bins=4, range=(0,1))[0]/(T.size)
        illumination = illumination_array.argmax()
        
        if illumination>0:
            vmax = self._weather2visibility[1] if self._weather2visibility[1]<=D.max() else D.max()
            if vmax<= self._weather2visibility[0]:
                visibility = self._weather2visibility[0]
            else:
                visibility = random.randint(self._weather2visibility[0], int(vmax))
            fog_color = random.randint(self._illumination2fogcolor[illumination][0], self._illumination2fogcolor[illumination][1])
            I_dark = reduce_lightHSV(I, sat_red=self._illumination2darkness[illumination], val_red=self._illumination2darkness[illumination])
            I_fog = fogAttenuation(I_dark, D, visibility=visibility, fog_color=fog_color)
        else:
            fog_color = 75
            visibility = 150 #D.max()*0.75
            I_fog = fogAttenuation(I, D, visibility=visibility, fog_color=fog_color)
            
        return I_fog

def main():
    args = parse_arguments()
    foggen = FogEffectGenerator()
    
    clearP = Path(args.clear_path)
    depthP = Path(args.depth_path)
    if clearP.is_file() and (depthP.is_file() and depthP.suffix==".npy"):
        snowy = foggen.genEffect(clearP, depthP)
        if args.show:
            Image.fromarray(snowy).show()

    if clearP.is_dir() and depthP.is_dir():
        if args.txt_file:
            with open(args.txt_file, 'r') as f:
                files = f.read().split('\n')
            image_files = [clearP / f for f in files]
        else:
            image_files = sorted(Path(clearP).glob("*"))

        depth_files = [Path(depthP) / ("-".join(imgf.name.split('-')[:2])+".npy") for imgf in image_files]
        
        valid_files = [idx for idx, f in enumerate(depth_files) if f.exists()]
        image_files = [image_files[idx] for idx in valid_files] 
        depth_files = [depth_files[idx] for idx in valid_files]
        
        save_folder = Path(args.save_folder)
        if not save_folder.exists():
            os.makedirs(str(save_folder))
        
        for imgp, depthp in tqdm(zip(image_files, depth_files), total=len(image_files)):
            foggy = foggen.genEffect(imgp, depthp)
            Image.fromarray(foggy).save(save_folder / (imgp.stem+"-fsyn.jpg"))

if __name__=='__main__':
    main()