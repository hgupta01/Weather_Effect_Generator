import os
from pydoc import visiblename
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from pathlib import Path
from skimage import io
import open3d.visualization as vis
#vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)

def calculateSnowDensity(R=1.5, intensity="regular"):
    ''' 
    input:
        R: snow fall rate in (mm/h)
        intensity = regular|storm
    output:
        N: snow density i.e. snow particles per m3 
    '''
    snow_intensity_2_rate = {"regular": 0.47, "storm": 0.3}
    k = snow_intensity_2_rate[intensity]
    Ms = k*R
    N = Ms/0.2
    return N

def getMedianSnowDia(R): return R**0.45

def calculateSnowDiaWeight(D, R):
    '''
    input:
        D: snow diameter (cm)
        R: snowfall rate (mm/h)
    output:
        Nd: weight of snow dia 
        
    #median diamter = D0 = 0.14*R^0.45 [cm], <1.3cm
    '''
    G = 22.9*(R**-0.45)
    N0 = 2.5*1e3*(R**-0.94)
    Nd = N0*np.exp(-G*D)
    return Nd

def weightedSnowflakeDiaEstimation(diameters, weights, Nflakes=100):
    """ 
    returns randomly a diameter from the sequence of dias with weighted likelihood
    input:
        - diameters: list of diameters (cm)
        - weights: list of diameter weights for likelihood calculation
        - Nflakes: number of diameters to be estimated
    output:
        - flake_dias: array of diameters (m)
    """

    weights = np.array(weights, dtype=np.float16)
    np.multiply(weights, 1 / weights.sum(), weights)
    weights = weights.cumsum()
    x = np.random.rand(Nflakes).astype(np.float16)
    inds = np.digitize(x, weights)
    inds[inds>=len(diameters)] = len(diameters)-1
    flake_dias = diameters[inds]/10
    return flake_dias


def main():
    snowfall_rate = 1.5
    snowfall_intensity = "regular"
    density_per_m3 = calculateSnowDensity(snowfall_rate, snowfall_intensity)

    snowflake_median_dia = 1.0 #getMedianSnowDia(snowfall_rate)
    snowfall_dia = np.zeros(15, dtype=np.float16)
    snowfall_dia[:8] = np.linspace(0.1, snowflake_median_dia, 8).astype(np.float16)
    snowfall_dia[7:] = np.linspace(snowflake_median_dia, 1.3, 8).astype(np.float16)
    snowfall_dia_weight = calculateSnowDiaWeight(snowfall_dia, snowfall_rate)

    edge_len = 15
    num_flakes = int(density_per_m3*edge_len**3)
    print(num_flakes)
    flakes_poses = np.random.rand(num_flakes, 3).astype(np.float16)*edge_len
    flakes_dia = weightedSnowflakeDiaEstimation(snowfall_dia, snowfall_dia_weight, Nflakes=num_flakes)
    print(len(flakes_dia))
    flakes = [o3d.geometry.TriangleMesh.create_sphere(radius=d*0.5, resolution=5) for d in flakes_dia]
    flakes = [f.translate(flakes_poses[i]) for i, f in enumerate(flakes)]
    for f in flakes:
        f.paint_uniform_color([0,0,0]) #np.random.randint(0, 55, 3)/255.0

    vis.draw(flakes)

if __name__ == '__main__':
    main()
