"""
Benchmark sul dataset Mirflickr 25K
"""

from pathlib import Path
from skimage.filters import threshold_multiotsu
from otsu import otsu_threshold_dp
import numpy as np
import cv2
import time

def benchmark_otsu(dataset_dir, classes=3, dataset_dim=25000):
    path = Path(dataset_dir)
    img_files = list(path.glob("*.jpg"))[:dataset_dim]
    total_images = len(img_files)
    
    results = {"prop" : [], "skimage": []}

    for i, img_path in enumerate(img_files, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue

        print(f"Processing {img_path.name} ({i}/{total_images})...")
        
        # Test per l'algoritmo proposto
        start_t = time.perf_counter()
        try:
            otsu_threshold_dp(img, classes)
            results["prop"].append(time.perf_counter() - start_t)
        except Exception as e:
            print(f"Errore proposto su {img_path.name}: {e}")
        
        # Test per skimage
        start_t = time.perf_counter()
        try:
            threshold_multiotsu(img, classes)
            results["skimage"].append(time.perf_counter() - start_t)
        except Exception as e:
            print(f"Errore skimage su {img_path.name}: {e}")
    
    return results

def print_stats(name, data_array):
    print(f"--- Statistiche per {name} ---")
    print(f"Media:     {np.mean(data_array):.6f} s")
    print(f"Mediana:   {np.median(data_array):.6f} s")
    print(f"Dev. Std:  {np.std(data_array):.6f} s")
    print(f"95° Perc.: {np.percentile(data_array, 95):.6f} s") 
    print(f"Tempo Tot: {np.sum(data_array):.2f} s")
    print("-" * 30)

if __name__ == "__main__":
    results = benchmark_otsu("mirflickr_25k/mirflickr", classes=3, dataset_dim=500)
    print_stats("Mio Otsu", np.array(results["prop"]))
    print_stats("Skimage Otsu", np.array(results["skimage"]))