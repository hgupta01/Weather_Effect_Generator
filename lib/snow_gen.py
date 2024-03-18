import random
import numpy as np
from lib.gen_utils import (generate_noisy_image, zoom_image_and_crop, get_otsu_threshold,
                           apply_motion_blur, color_level_adjustment, repeat_and_combine, crystallize,
                           layer_blend)


class SnowGenUsingNoise:
    def __init__(self):
        self._noise_scale_range = {
            # 'small': (0.24, 0.45), 'large': (0.45, 0.65)
            'small': (0.1, 0.2), 'large': (0.3, 0.5)
            }
        self._noise_amount_range = {
            # 'small': (0.35, 0.65), 'large': (0.05, 0.15)
            'small': (0.25, 0.45), 'large': (0.05, 0.15)
            }
        # self._zoom_range = {'small': (1.75, 3.0), 'large': (7, 10)}
        self._zoom_range = {'small': (1.5, 2.0), 'large': (4, 6)}
        self._blur_kernel_range = {'small': [3, 5, 7], 'large': [9, 11, 13]}
        self._repeat_scale = {'small': [0], 'large': [0]}
        self._max_level = {'small': (100, 150), 'large': (200, 250)}
        self._cyrstalize_range = (0.55, 0.75)

    def genSnowLayer(self,
                     h,
                     w,
                     noise_scale=0.5,
                     noise_amount=0.25,
                     zoom_layer=2.0,
                     blur_kernel_size=15,
                     blur_angle=-60,
                     max_level=250,
                     compress_scale=0,
                     cyrstalize_amount=0.5
                     ):
        im_noisy = generate_noisy_image(
            h, w, sigma=noise_scale, p=noise_amount)
        im_zoom = zoom_image_and_crop(im_noisy, r=zoom_layer)
        im_blurr = apply_motion_blur(im_zoom, blur_kernel_size, blur_angle)

        ret = get_otsu_threshold(im_blurr)
        layer = color_level_adjustment(
            im_blurr.copy(), inBlack=ret, inWhite=max_level, inGamma=1.0)

        if compress_scale > 0:
            layer = repeat_and_combine(layer, compress_scale)

        if cyrstalize_amount > 0:
            layer = crystallize(np.flipud(layer), r=0.75)

        return layer.astype(np.uint8)

    def genSnowMultiLayer(self, h, w, blur_angle=75, intensity="large", num_itr=2):
        noise_scale_range = self._noise_scale_range[intensity]
        noise_amount_range = self._noise_amount_range[intensity]
        zoom_range = self._zoom_range[intensity]
        blur_kernel_range = self._blur_kernel_range[intensity]
        repeat_scale = self._repeat_scale[intensity][0]
        max_level = self._max_level[intensity]

        layer = np.zeros((h, w), dtype=np.uint8)
        for _ in range(num_itr):
            l = self.genSnowLayer(h, w,
                                  noise_scale=random.uniform(
                                      noise_scale_range[0], noise_scale_range[1]),
                                  noise_amount=random.uniform(
                                      noise_amount_range[0], noise_amount_range[1]),
                                  zoom_layer=random.uniform(
                                      zoom_range[0], zoom_range[1]),
                                  blur_kernel_size=random.choice(
                                      blur_kernel_range),
                                  blur_angle=blur_angle,
                                  max_level=random.randint(
                                      max_level[0], max_level[1]),
                                  compress_scale=repeat_scale,
                                  cyrstalize_amount=random.uniform(
                                      self._cyrstalize_range[0], self._cyrstalize_range[1])
                                  )
            # tr = 0.25 + random.random()*0.5
            # layer = layer + tr*l  
            layer = layer_blend(layer, l)
        return layer
