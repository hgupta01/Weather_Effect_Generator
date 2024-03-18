import os
import copy
import torch
import random
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from lib.style_transfer_utils import (tensor2pil, 
                                      load_style_transfer_model, 
                                      run_style_transfer,
                                      style_content_image_loader)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-imgs", type=str, help="Path to the content images.", required=True)
    parser.add_argument("--style-imgs", type=str, help="Path to the style images.", required=True)
    parser.add_argument("--save-folder", type=str, help="Path to the save the generated images.", required=True)
    parser.add_argument("--vgg", type=str, help="Path to the pretrained VGG model.", required=True)

    parser.add_argument('--cuda', action='store_true', help="use cuda.")
    parser.add_argument('--ext', type=str, default="stl", help="extension for generated image.")
    parser.add_argument('--min-step', type=int, default=100, help="minimum iteration steps")
    parser.add_argument('--max-step', type=int, default=200, help="maximum iteration steps")
    parser.add_argument('--style-weight', type=float, default=100000, help="weight for style loss")
    parser.add_argument('--content-weight', type=float, default=2, help="weight for content loss")

    return parser.parse_args()


def transfer_style(cnn_path, 
                   cimg, 
                   simg,
                   min_step=100, 
                   max_step=200,
                   style_weight=100000,
                   content_weight=2,
                   device="cpu"
                  ):
    cnn = load_style_transfer_model(pretrained=cnn_path)
    
    content_img, style_img = style_content_image_loader(cimg, simg)
    input_img = copy.deepcopy(content_img).to(device, torch.float)

    output = run_style_transfer(cnn, 
                                content_img,
                                style_img,
                                input_img,  
                                num_steps=random.randint(min_step, max_step),
                                style_weight=style_weight, 
                                content_weight=content_weight,
                                device=device)
    return tensor2pil(output[0].detach().cpu())


def main():
    args = parse_arguments()
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # content_images = sorted(Path(args.content_imgs).glob("*"))
    with open(args.content_imgs, "r") as f:
        lines = f.read()
    content_images = lines.split('\n')
    content_images = [Path("/home/saikrishna/ML_Projetcs/Datasets/Adverse_Weather_Simulation/test/clear") / f for f in content_images]
    style_images = sorted(Path(args.style_imgs).glob("*"))

    save_folder = Path(args.save_folder)
    if not os.path.exists(args.save_folder):
        print(f"Creating {args.save_folder}")
        os.makedirs(str(save_folder))

    for cimg in tqdm(content_images):
        name, extension = cimg.name.split('.')
        simg = random.choice(style_images)

        output_img = transfer_style(cnn_path=args.vgg, 
                                    cimg=cimg, 
                                    simg=simg, 
                                    min_step=args.min_step, 
                                    max_step=args.max_step,
                                    style_weight=args.style_weight, 
                                    content_weight=args.content_weight,
                                    device=device)
        output_img.save(save_folder / f"{name}-{args.ext}.{extension}")
        
if __name__ == '__main__':
    main()
