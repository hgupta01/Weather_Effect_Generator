import cv2
import numpy as np
from skimage import measure
from skimage import color, filters
from sklearn.neighbors import NearestNeighbors


def get_otsu_threshold(image):
    image = cv2.GaussianBlur(image.astype(float), (7, 7), 0)
    ret, _ = cv2.threshold(image.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret


def reduce_lightHSV(rgb, sat_red=0.5, val_red=0.5):
        hsv = color.rgb2hsv(rgb/255)
        hsv[...,1] *= sat_red
        hsv[...,2] *= val_red
        return (color.hsv2rgb(hsv)*255).astype(np.uint8)


def apply_motion_blur(image, size, angle):
    '''
    input:
        image - numpy array of image
        size - in pixels, size of motion blur
        angel - in degrees, direction of motion blur
    output:
        blurred image as numpy array
    '''
    k = np.zeros((size, size), dtype=np.float32)
    k[(size-1) // 2, :] = np.ones(size, dtype=np.float32)
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D(
        (size / 2 - 0.5, size / 2 - 0.5), angle, 1.0), (size, size))
    k = k * (1.0 / np.sum(k))
    return cv2.filter2D(image, -1, k).astype(np.uint8)


def illumination2opacity(img: np.ndarray, illumination):
        alpha = color.rgb2gray(img)
        if illumination>0:
            alpha = np.clip(filters.gaussian((1-alpha), sigma=20, channel_axis=None),0,1)
        else:
            alpha = np.clip(2*filters.gaussian((alpha), sigma=20, channel_axis=None),0,1)
        return alpha
    

def color_level_adjustment(image, inBlack=0, inWhite=255, inGamma=1.0, outBlack=0, outWhite=255):
    '''
    Adjust color level.
    input:
        image    - numpy array of greyscale image
        inBlack  - lower limit of intensity
        inWhite  - upper limit of intensity
        inGamma  - scaling the intensity values by Gamma value
        outBlack - lower intensity value for scaling 
        outWhite - upper intensity value for scaling 
    '''
    assert image.ndim == 2

    # image = np.clip( (image - inBlack) / (inWhite - inBlack), 0, 1)
    image = (image - inBlack) / (inWhite - inBlack)
    image[image < 0] = 0
    image[image > 1] = 0
    image = (image ** (1/inGamma)) * (outWhite - outBlack) + outBlack
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def crystallize(img, r):
    '''
    Crystallization Effect
    input: img - Numpy Array 
           r   - fraction of  pixels to select as center for crystallization 
    outpur: res- Numpy Array for crystallized filter
    '''
    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w, _ = img.shape

    # Get the center for crystallization
    pixels = np.zeros((h*w, 2), dtype=np.uint16)
    pixels[:, 0] = np.tile(np.arange(h), (w, 1)).T.reshape(-1)
    pixels[:, 1] = (np.tile(np.arange(w), (h, 1))).reshape(-1)

    sel_pixels = pixels.copy()
    sel_pixels = sel_pixels[np.random.randint(0, h*w, int(len(sel_pixels)*r))]

    # Perform nearest neighbour for all pixels
    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='ball_tree', n_jobs=4).fit(sel_pixels)
    distances, indices = nbrs.kneighbors(pixels)
    color_pixels = sel_pixels[indices[:, 0]]

    # Perform crystallization (copy the color pixels of crystal center)
    res = np.zeros_like(img)
    res[pixels[:, 0], pixels[:, 1]] = img[color_pixels[:, 0], color_pixels[:, 1]]
    return res


def zoom_image_and_crop(image, r=1.5):
    '''
    input: 
        image: numpy array
        r = upscale fraction >1.0
    output:
        image: scale image as numpy array
    '''
    if image.ndim == 2:
        h, w = image.shape
    elif image.ndim == 3:
        h, w, _ = image.shape
    image_resize = cv2.resize(image.astype(np.uint8), (int(
        w*r), int(h*r)), interpolation=cv2.INTER_LANCZOS4)

    x = int(r*w/2 - w/2)
    y = int(r*h/2 - h/2)
    crop_img = image_resize[int(y):int(y+h), int(x):int(x+w)]

    return crop_img.astype(np.uint8)


def repeat_and_combine(layer, repeat_scale=2):
    orgh, orgw = layer.shape
    compressh = int(np.floor(orgh/repeat_scale))
    compressw = int(np.floor(orgw/repeat_scale))

    resize_layer = cv2.resize(
        layer, (compressw, compressh), interpolation=cv2.INTER_LANCZOS4)
    layer_tile = np.tile(resize_layer, (repeat_scale, repeat_scale))
    h, w = layer_tile.shape

    repeat = np.zeros_like(layer)
    repeat[:h, :w] = layer_tile
    return repeat.astype(np.uint8)


def generate_noisy_image(h, w, sigma=0.5, p=0.5):
    '''
    input: 
        h - height of the image
        w - width of the image
        scale - scale of Gaussian noise
    output:
        im_noisy - uint8 array with Gaussian noise
    '''
    im_array = np.zeros((h, w))

    # Generate random Gaussian noise
    noise = np.random.normal(scale=sigma, size=(h, w))
    prob = np.random.rand(h, w)
    im_array[prob < p] = 255*noise[prob < p]
    im_array = np.clip(im_array, 0, 255)
    return im_array.astype(np.uint8)


def binarizeImage(image: np.ndarray):
    """Binarize grey image using OTSU threshold"""
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image = image[:, :, 0]
    binarize = np.copy(image)
    ret = get_otsu_threshold(image=image)
    binarize[binarize < ret] = 0
    binarize[binarize > ret] = 255
    return binarize


def bwAreaFilter(mask, area_range=(0, np.inf)):
    """Extract objects from binary image by size """
    labels = measure.label(mask.astype('uint8'), background=0)
    unq, areas = np.unique(labels, return_counts=True)
    areas = areas[1:]
    area_idx = np.arange(1, np.max(labels) + 1)

    inside_range_idx = np.logical_and(
        areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    layer = np.isin(labels, area_idx)
    return layer.astype(int)


def centreCrop(image, reqH, reqW):
    center = image.shape
    x = center[1]/2 - reqW/2
    y = center[0]/2 - reqH/2

    crop_img = image[int(y):int(y+reqH), int(x):int(x+reqW)]
    return crop_img


def alpha_blend(img, layer, alpha):
    if layer.ndim == 3:
        layer = cv2.cvtColor(layer.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    assert alpha.ndim == 2
    assert layer.ndim == 2
    blended = img*(1-alpha[:, :, None]) + layer[:, :, None]*alpha[:, :, None]
    return blended


def screen_blend(image, layer):
    '''
    input:
        image - numpy array of RGB image
        layer - numpy array of layer to blend
    '''
    result = 255.0*(1 - (1-image/255.0)*(1-layer[:, :, None]/255.0))
    return result.astype(np.uint8)


def layer_blend(layer1, layer2):
    '''
    input:
        layer1 - numpy array of RGB image
        layer2 - numpy array of layer to blend
    '''
    assert layer1.shape == layer2.shape
    result = 255.0*(1 - (1-layer1/255.0)*(1-layer2/255.0))
    return result.astype(np.uint8)


def scale_depth(im, nR, nC):
    nR0 = len(im)     # source number of rows 
    nC0 = len(im[0])  # source number of columns 
    return np.asarray([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]
            for c in range(nC)] for r in range(nR)])