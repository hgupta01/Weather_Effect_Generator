import numpy as np
from PIL import Image
from noise import pnoise3

def perlin_noise(w, h, depth):
    p1 = Image.new('L', (w, h))
    p2 = Image.new('L', (w, h))
    p3 = Image.new('L', (w, h))

    scale = 1/130.0
    for y in range(h):
        for x in range(w):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+1)*128.0)
            p1.putpixel((x, y), color)

    scale = 1/60.0
    for y in range(h):
        for x in range(w):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+0.5)*128)
            p2.putpixel((x, y), color)

    scale = 1/10.0
    for y in range(h):
        for x in range(w):
            v = pnoise3(x * scale, y * scale, depth[y, x] * scale, octaves=1, persistence=0.5, lacunarity=2.0)
            color = int((v+1.2)*128)
            p3.putpixel((x, y), color)

    perlin = (np.array(p1) + np.array(p2)/2 + np.array(p3)/4)/3

    return perlin

def generate_fog(image, depth, visibility=None, fog_color=None):
    '''
    input:
        image - numpy array (h, w, c) 
        depth - numpy array (h, w)
    '''

    height, width  = depth.shape
    perlin = perlin_noise(width, height, depth)

    depth_max = depth.max()
    
    if visibility:
        fog_visibility = visibility
    else:
        fog_visibility = float(np.random.randint(int(depth_max-0.2*depth_max), int(depth_max+0.2*depth_max)) )
        fog_visibility = np.clip(fog_visibility, 60, 200)

    VERTICLE_FOV = 60 #degrees
    CAMERA_ALTITUDE = 1.8 #meters
    VISIBILITY_RANGE_MOLECULE = 12  # m    12
    VISIBILITY_RANGE_AEROSOL = fog_visibility  # m     450
    ECM_ = 3.912 / VISIBILITY_RANGE_MOLECULE  # EXTINCTION_COEFFICIENT_MOLECULE /m
    ECA_ = 3.912 / VISIBILITY_RANGE_AEROSOL  # EXTINCTION_COEFFICIENT_AEROSOL /m


    FT = 70  # FOG_TOP m  31  70
    HT = 34  # HAZE_TOP m  300    34

    angle = np.repeat(-1*np.linspace(-0.5*VERTICLE_FOV, 0.5*VERTICLE_FOV, height).reshape(-1,1), axis=1, repeats=width)
    distance = depth / np.cos(np.radians(angle))
    elevation = CAMERA_ALTITUDE + distance * np.sin(np.radians(angle))

    distance_through_fog = np.zeros_like(distance)
    distance_through_haze = np.zeros_like(distance)
    distance_through_haze_free = np.zeros_like(distance)

    ECA = ECA_
    c = 1 - elevation / (FT + 0.00001)
    c[c < 0] = 0
    ECM = (ECM_ * c + (1 - c) * ECA_) * (perlin / 255)

    idx1 = np.logical_and(FT > elevation, elevation > HT)
    idx2 = elevation <= HT
    idx3 = elevation >= FT

    distance_through_haze[idx2] = distance[idx2]
    distance_through_fog[idx1] = (
        (elevation[idx1] - HT)
        * distance[idx1]
        / (elevation[idx1] - CAMERA_ALTITUDE)
    )
    distance_through_haze[idx1] = distance[idx1] - distance_through_fog[idx1]
    distance_through_haze[idx3] = (
        (HT - CAMERA_ALTITUDE)
        * distance[idx3]
        / (elevation[idx3] - CAMERA_ALTITUDE)
    )
    distance_through_fog[idx3] = (
        (FT - HT)
        * distance[idx3]
        / (elevation[idx3] - CAMERA_ALTITUDE)
    )
    distance_through_haze_free[idx3] = (
        distance[idx3] - distance_through_haze[idx3] - distance_through_fog[idx3]
    )

    attenuation = np.exp(-ECA * distance_through_haze - ECM * distance_through_fog)

    I_ex = image * attenuation[:,:,None]
    O_p = 1 - attenuation
    if fog_color is None:
        fog_color = np.random.randint(200,255)
    I_al = np.array([[[fog_color, fog_color, fog_color]]])

    I = I_ex + O_p[:,:,None] * I_al
    return I.astype(np.uint8)

def fogAttenuation(img: np.ndarray, depth:np.ndarray, visibility=1000, fog_color=200):
        img_fog = generate_fog(img.copy(), depth.copy(), visibility=visibility, fog_color=fog_color)
        return img_fog