"""
Benchmark sul dataset Mirflickr 25K
Confronto tra algoritmo Otsu proposto e skimage multi-Otsu
"""

from pathlib import Path
from skimage.filters import threshold_multiotsu
from otsu import otsu_threshold_dp
import numpy as np
import cv2
import time
import random
import json
from scipy import stats
import csv
from datetime import datetime

# ──────────────────────────────────────────────
# Configurazione
# ──────────────────────────────────────────────

SEED        = 42
WARMUP_IMGS = 3      # immagini di warm-up escluse dalle statistiche
REPEATS     = 3      # ripetizioni per immagine (si prende la mediana)
ATOL        = 1      # tolleranza assoluta per confronto soglie (livelli di grigio)


# ──────────────────────────────────────────────
# Funzioni di misurazione
# ──────────────────────────────────────────────

def measure(fn, img, repeats=REPEATS):
    """
    Esegue `fn(img)` per `repeats` volte e restituisce
    (mediana dei tempi, risultato dell'ultima chiamata).
    """
    times  = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(img)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def run_pair(img, classes, rng):
    """
    Esegue entrambi gli algoritmi su una singola immagine in ordine casuale
    per evitare vantaggi sistematici da cache.

    Restituisce un dict con tempi, soglie e flag di validità,
    oppure None se uno dei due algoritmi fallisce.
    """
    algorithms = [
        ("prop",    lambda im: otsu_threshold_dp(im, classes)[0]),
        ("skimage", lambda im: threshold_multiotsu(im, classes)),
    ]
    rng.shuffle(algorithms)

    record = {}
    for name, fn in algorithms:
        try:
            elapsed, thresholds = measure(fn, img)
            record[name] = {
                "time": elapsed,
                "thresholds": np.sort(np.asarray(thresholds, dtype=np.float64))
            }
        except Exception as e:
            print(f"    ⚠  Errore [{name}]: {e}")
            return None

    return record


# ──────────────────────────────────────────────
# Warm-up
# ──────────────────────────────────────────────

def warmup(img_files, classes, n=WARMUP_IMGS):
    """
    Esegue entrambi gli algoritmi su `n` immagini senza registrare i tempi,
    per stabilizzare JIT, import e cache OS prima del benchmark reale.
    """
    print(f"[Warm-up] {n} immagini escluse dalle statistiche...")
    for img_path in img_files[:n]:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            otsu_threshold_dp(img, classes)
            threshold_multiotsu(img, classes)
        except Exception:
            pass
    print("[Warm-up] completato.\n")


# ──────────────────────────────────────────────
# Controllo correttezza
# ──────────────────────────────────────────────

def check_correctness(record, img_name):
    """
    Confronta le soglie dei due algoritmi e stampa un avviso
    se divergono oltre la tolleranza `ATOL`.
    """
    t_prop = np.sort(np.asarray(record["prop"]["thresholds"], dtype=np.float64))
    t_sk   = np.sort(np.asarray(record["skimage"]["thresholds"], dtype=np.float64))
    if not np.allclose(t_prop, t_sk, atol=ATOL):
        print(f"    ⚠  Soglie divergenti su {img_name}: prop={t_prop} | skimage={t_sk}")


# ──────────────────────────────────────────────
# Benchmark principale
# ──────────────────────────────────────────────

def benchmark_otsu(dataset_dir, classes=3, dataset_dim=25000):
    rng  = random.Random(SEED)
    path = Path(dataset_dir)

    all_files = list(path.glob("*.jpg"))
    rng.shuffle(all_files)
    img_files = all_files[:dataset_dim + WARMUP_IMGS]

    warmup(img_files, classes, n=WARMUP_IMGS)

    benchmark_files = img_files[WARMUP_IMGS:]
    total = len(benchmark_files)

    times = {"prop": [], "skimage": []}

    for i, img_path in enumerate(benchmark_files, 1):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Impossibile leggere {img_path.name}, saltata.")
            continue

        print(f"[{i:>5}/{total}] {img_path.name}")
        record = run_pair(img, classes, rng)

        if record is None:
            continue

        check_correctness(record, img_path.name)

        times["prop"].append(record["prop"]["time"])
        times["skimage"].append(record["skimage"]["time"])

    return {k: np.array(v) for k, v in times.items()}


# ──────────────────────────────────────────────
# Statistiche e analisi
# ──────────────────────────────────────────────

def print_stats(name, data):
    print(f"  Algoritmo : {name}")
    print(f"  N         : {len(data)}")
    print(f"  Media     : {np.mean(data):.6f} s")
    print(f"  Mediana   : {np.median(data):.6f} s")
    print(f"  Dev. Std  : {np.std(data):.6f} s")
    print(f"  95° Perc. : {np.percentile(data, 95):.6f} s")
    print(f"  Tempo tot : {np.sum(data):.2f} s")


def analyze(results):
    prop   = results["prop"]
    sk     = results["skimage"]

    print("=" * 45)
    print_stats("Otsu proposto", prop)
    print("-" * 45)
    print_stats("Skimage Otsu",  sk)
    print("=" * 45)

    # Speedup
    speedup = np.median(sk) / np.median(prop)
    faster  = "proposto" if speedup >= 1 else "skimage"
    print(f"\n  Speedup mediano ({faster} più veloce): {abs(speedup):.2f}×")

    # Test di significatività (Wilcoxon signed-rank, paired, non-parametrico)
    if len(prop) >= 10:
        stat, p = stats.wilcoxon(prop, sk, alternative="two-sided")
        sig     = "OK differenza statisticamente significativa (p < 0.05)" if p < 0.05 \
                  else "! differenza NON significativa (p ≥ 0.05)"
        print(f"  Wilcoxon p-value: {p:.4e}  →  {sig}")
    else:
        print("  (campione troppo piccolo per il test di Wilcoxon)")

    print("=" * 45)


# ──────────────────────────────────────────────
# Salvataggio risultati
# ──────────────────────────────────────────────

def save_results(results, path="results_otsu.npz"):
    np.savez(path, prop=results["prop"], skimage=results["skimage"])
    print(f"\n  Risultati raw salvati in '{path}'")
    print(f"  Per ricaricarli: data = np.load('{path}')")

def save_results_csv(results, classes, dataset_dim, path="results_otsu.csv"):
    prop = results["prop"]
    sk   = results["skimage"]

    row = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "classes":          classes,
        "dataset_dim":      dataset_dim,
        "n_valid":          len(prop),
        "prop_mean":        round(float(np.mean(prop)),   6),
        "prop_median":      round(float(np.median(prop)), 6),
        "prop_std":         round(float(np.std(prop)),    6),
        "prop_p95":         round(float(np.percentile(prop, 95)), 6),
        "prop_total":       round(float(np.sum(prop)),    4),
        "skimage_mean":     round(float(np.mean(sk)),     6),
        "skimage_median":   round(float(np.median(sk)),   6),
        "skimage_std":      round(float(np.std(sk)),      6),
        "skimage_p95":      round(float(np.percentile(sk, 95)), 6),
        "skimage_total":    round(float(np.sum(sk)),      4),
        "speedup_median":   round(float(np.median(sk) / np.median(prop)), 4),
    }

    file_path = Path(path)
    write_header = not file_path.exists()

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n  Risultati CSV {'creato' if write_header else 'aggiornato'}: '{path}'")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    CLASSES = 3
    DATASET_DIM = 50

    results = benchmark_otsu(
        dataset_dir = "mirflickr_25k/mirflickr",
        classes = CLASSES,
        dataset_dim = DATASET_DIM,
    )

    print("\n")
    analyze(results)
    save_results(results)
    save_results_csv(results, classes=CLASSES, dataset_dim=DATASET_DIM)