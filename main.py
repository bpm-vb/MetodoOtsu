"""
=======================
Multi-Otsu Thresholding
=======================

Confrontiamo l'algoritmo di Scikit con quello proposto
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from otsu import otsu_threshold_dp

from skimage import data
from skimage.filters import threshold_multiotsu

matplotlib.rcParams['font.size'] = 9

def compute_otsu(image: any, classes: int):
    a = time.perf_counter()
    thresholds_my, _ = otsu_threshold_dp(image, classes)
    print(f"Proposed: {time.perf_counter()-a:.4f}\tThresholds: {list(thresholds_my)}")
    a = time.perf_counter()
    thresholds = threshold_multiotsu(image, classes)
    print(f"Scikit: {time.perf_counter()-a:.4f}\tThresholds: {list(thresholds)}")

    regions = np.digitize(image, bins=thresholds)
    regions_my = np.digitize(image, bins=thresholds_my)

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(18, 3.5))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(image.ravel(), bins=255)
    ax[1].set_title('Histogram')
    for thresh in thresholds:
        ax[1].axvline(thresh, color='r')

    ax[2].imshow(regions, cmap='jet')
    ax[2].set_title('Scikit Multi-Otsu result')
    ax[2].axis('off')

    ax[3].hist(image.ravel(), bins=255)
    ax[3].set_title('Histogram')
    for thresh in thresholds_my:
        ax[3].axvline(thresh, color='r')

    ax[4].imshow(regions_my, cmap='jet')
    ax[4].set_title('Proposed Multi-Otsu result')
    ax[4].axis('off')

    plt.subplots_adjust()

    plt.show()

if __name__=="__main__":
    image = data.camera()
    compute_otsu(image, 6)