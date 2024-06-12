import os
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True, help="path to the file or the folder")
    parser.add_argument("--save_folder", type=str, default="./depth/", help="path to the folder")
    parser.add_argument("--midas_model", type=str, default="DPT_Large", help="Midas model name")
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--baseline", type=float, default=0.54)
    parser.add_argument("--focal", type=float, default=721.09)
    parser.add_argument("--img_scale", type=float, default=1)
    return parser.parse_args()


def get_depth_estimation_model(model_name:str, device="cpu"):
    assert model_name in ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]
    
    midas = torch.hub.load("intel-isl/MiDaS", model_name)
    midas.eval()
    midas.to(device)
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_name in ["DPT_Large", "DPT_Hybrid"]:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform
    return midas, transform


def getDisparityMap(model, transform, img_path):
    img = Image.open(img_path)

    input_batch = transform(img)
    with torch.no_grad():
        prediction = model(input_batch.cuda())

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()


def main():
    args = parse_arguments()
    
    device = torch.device("cpu") 
    if args.use_cuda:
        device = torch.device("cuda") 
        
    ### kitti
    baseline = args.baseline
    focal = args.focal
    img_scale = args.img_scale
    
    imgP = Path(args.img_path)
    save_folder = Path(args.save_folder)
    if not save_folder.exists():
        os.makedirs(str(save_folder))
    
    midas, midas_transform = get_depth_estimation_model(model_name=args.midas_model, device=device)
    
    if imgP.is_file():
        disp = getDisparityMap(midas, midas_transform, imgP)
        disp[disp<0]=0
        disp = disp + 1e-3
        depth = baseline*focal/(disp*img_scale)
        np.save(save_folder / imgP.stem, depth)

    if imgP.is_dir():
        image_files = sorted(Path(imgP).glob("*"))
        for imgp in tqdm(image_files):
            disp = getDisparityMap(midas, midas_transform, imgp)
            disp[disp<0]=0
            disp = disp + 1e-3
            depth = baseline*focal/(disp*img_scale)
            np.save(save_folder / imgP.stem, depth)
            

if __name__=='__main__':
    main()
