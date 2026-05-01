import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------------------------
# Implementazione di OTSU
# ---------------------------------------------------------------------------

def compute_histogram(gray: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Ritorna l'istogramma dell'immagine con 256 bin (0-255)"""
    hist = np.zeros(256, dtype=np.float64)
    for val in gray.ravel():
        hist[val] += 1
    if normalize:
        hist /= hist.sum()          # probabilità p(i)
    return hist

def otsu_threshold(gray: np.ndarray) -> tuple[int, float]:
    """
    Calcola la soglia ottimale seguendo l'algoritmo di Otsu.

    Idea:
      Per ogni soglia t in [0, 254] si dividono i pixel in due classi:
        C0 = {0..t},  C1 = {t+1..255}
      Si massimizza la varianza inter-classe:
        sigma_b_sq(t) = omega_0(t)·omega_1(t)·[mu_0(t) - mu_1(t)]²
      che è equivalente a:
        sigma_b_sq(t) = [mu_T·omega_0(t) - mu_0_cum(t)]² / [omega_0(t)·(1-omega_0(t))]

    Returns
    -------
    threshold : int
        Soglia ottimale [0,255].
    sigma_max : float
        Valore massimo della varianza inter-classe.
    """
    hist = compute_histogram(gray, normalize=True)   # p(i)
    levels = np.arange(256, dtype=np.float64)

    # Calcolo dei valori cumulativi
    omega = np.cumsum(hist)                          # P(C0) per le possibili soglie
    mu_cum = np.cumsum(hist * levels)                # Media cumulativa per le possibili soglie
    mu_T = mu_cum[-1]                                # Media globale

    # Varianza inter-classe per tutte le possibili soglie in una sola riga vettorizzata
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = np.where(
            (omega > 0) & (omega < 1),
            (mu_T * omega - mu_cum) ** 2 / (omega * (1.0 - omega)),
            0.0
        )

    threshold = int(np.argmax(sigma_b_sq))
    sigma_max = float(sigma_b_sq[threshold])
    return threshold, sigma_max

def class_contribution(a, b, omega, mu_cum, mu_T):
    """
    Calcola quanto una certa classe [a, b] contribuisce alla varianza inter-classe

    Idea:
      Si calcola la probabilità (% di pixel appartenti) di una classe (omega_ab)
      e la sua media cumulativa (mu_ab_cum).
      Poi si calcola l'intensità media (mu_ab) dei pixel della classe [a, b] e
      si ritorna il risultato del contributo statistico:
        omega_ab * (mu_ab - mu_T)**2

    Returns
    -------
    sigma_b_sq_ab : float
        Il risultato del contributo della classe [a, b] sulla varianza interclasse sigma_b_sq
    """
    if a == 0:
        omega_ab = omega[b]
        mu_ab_cum = mu_cum[b]
    else:
        omega_ab = omega[b] - omega[a-1]
        mu_ab_cum = mu_cum[b] - mu_cum[a-1]

    if omega_ab == 0:
        return 0.0

    mu_ab = mu_ab_cum / omega_ab
    sigma_b_sq_ab = omega_ab * (mu_ab - mu_T)**2
    return float(sigma_b_sq_ab)

def otsu_threshold_dp(gray: np.ndarray, k: int) -> tuple[list[int], float]:
    
    # Istogramma e cumulativi
    hist = compute_histogram(gray, normalize=True)   # p(i)
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(hist)                          # P(C0) per le possibili soglie
    mu_cum = np.cumsum(hist * levels)                # Media cumulativa per le possibili soglie
    mu_T = mu_cum[-1]                                # Media globale

    # Tabella con soglie e livelli
    variance_table = np.zeros((k+1, 256))
    thresholds = np.zeros((k+1, 256))
    best_thresholds = []

    for t in range(256): # caso base in cui utilizzo 0 soglie
        variance_table[0][t] = class_contribution(0, t, omega, mu_cum, mu_T)

    for i in range(1, k+1): # utilizzo le soglie fino a k
        for t in range(i, 256): # itero sui livelli, da 0 a [i, 256], parto da i perché così posso calcolare le mie k=i soglie altrimenti non ci riuscirei avendo un array più piccolo
            best = -np.inf
            best_threshold = 0

            for t_k in range(i-1, t): # possibile ultima soglia
                if variance_table[i-1][t_k] == 0 and t_k != i-1:
                    continue
                curr = variance_table[i-1][t_k] + class_contribution(t_k+1, t, omega, mu_cum, mu_T)
                if curr > best: 
                    best = curr
                    best_threshold = t_k
            variance_table[i][t] = best
            thresholds[i][t] = best_threshold

    t = 255
    for i in range(k, 0, -1):
        t = int(thresholds[i][t])
        best_thresholds.append(t)
    best_thresholds.reverse()

    return best_thresholds, variance_table[k][255]

def apply_threshold(gray: np.ndarray, t: int) -> np.ndarray:
    """Binarizza l'immagine con soglia t (pixel > t → 255, altrimenti 0)."""
    return np.where(gray > t, 255, 0).astype(np.uint8)

def apply_threshold_multi(gray: np.ndarray, soglie: list[int]) -> np.ndarray:
    colori = [
        (70,  130, 180),   # blu acciaio
        (220,  80,  80),   # rosso
        (80,  180,  80),   # verde
        (220, 160,  40),   # arancio
        (160,  80, 200),   # viola
    ]

    classi = np.digitize(gray, bins=soglie)  # indice classe per ogni pixel
    output = np.zeros((*gray.shape, 3), dtype=np.uint8)
    for idx, colore in enumerate(colori[:len(soglie) + 1]):
        output[classi == idx] = colore

    return output


# ---------------------------------------------------------------------------
# Implementazione di OTSU con OpenCV
# ---------------------------------------------------------------------------

def otsu_opencv(gray: np.ndarray) -> tuple[int, np.ndarray]:
    """Soglia Otsu tramite OpenCV (reference)."""
    t_cv, binary = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t_cv), binary


# ---------------------------------------------------------------------------
# Caricamento immagini di test
# ---------------------------------------------------------------------------

def load_sample_images() -> dict[str, np.ndarray]:
    paths = {
        "persona.jpg":   r"persona.png",
        "edificio.jpg":  r"edificio.png",
    }

    images = {}
    for name, path in paths.items():
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images[name] = img
        else:
            print(f"  [✗] {name}: file non trovato")
    return images


# ---------------------------------------------------------------------------
# Visualizzazione dei risultati
# ---------------------------------------------------------------------------

def plot_results(images: dict[str, np.ndarray], k: int = 2) -> None:
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Otsu Thresholding  —  single vs multi ({k} soglie)",
                 fontsize=15, fontweight="bold", y=1.01)

    for row, (name, gray) in enumerate(images.items()):
        ax_img, ax_hist, ax_single, ax_multi = axes[row]

        # --- Otsu singola soglia ---
        t_our, sigma_max = otsu_threshold(gray)
        binary_single = apply_threshold(gray, t_our)

        # --- Otsu multi soglia DP ---
        soglie, sigma_dp = otsu_threshold_dp(gray, k)
        binary_multi = apply_threshold_multi(gray, soglie)

        # Immagine originale
        ax_img.imshow(gray, cmap="gray", vmin=0, vmax=255)
        ax_img.set_title(f"{name}\n(originale)", fontsize=9)
        ax_img.axis("off")

        # Istogramma con tutte le soglie
        hist = compute_histogram(gray, normalize=True)
        ax_hist.bar(range(256), hist, color="steelblue", width=1, alpha=0.7)
        ax_hist.axvline(t_our, color="white", lw=2, ls="--", label=f"single t={t_our}")
        colori_hist = ["crimson", "limegreen", "gold", "darkorange", "violet"]
        for idx, s in enumerate(soglie):
            ax_hist.axvline(s, color=colori_hist[idx % len(colori_hist)],
                            lw=1.5, label=f"t{idx+1}={s}")
        ax_hist.set_title("Istogramma + soglie", fontsize=9)
        ax_hist.legend(fontsize=7)
        ax_hist.set_xlim(0, 255)

        # Singola soglia
        ax_single.imshow(binary_single, cmap="gray", vmin=0, vmax=255)
        ax_single.set_title(f"Single  t={t_our}\nσ²_B={sigma_max:.1f}", fontsize=9)
        ax_single.axis("off")

        # Multi soglia
        ax_multi.imshow(binary_multi)
        ax_multi.set_title(f"Multi k={k}  soglie={soglie}\nσ²_B={sigma_dp:.1f}", fontsize=9)
        ax_multi.axis("off")

    plt.tight_layout()
    out_path = Path("otsu_results.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[✓] Figure salvata in {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Report testuale
# ---------------------------------------------------------------------------

def print_report(images: dict[str, np.ndarray], k: int = 2) -> None:
    print("\n" + "=" * 75)
    print(f"{'Immagine':<28} {'t_otsu':>8} {'t_opencv':>9} {'soglie_dp':>20} {'σ²_B':>12}")
    print("-" * 75)
    for name, gray in images.items():
        t_our, sigma_max = otsu_threshold(gray)
        t_cv, _ = otsu_opencv(gray)
        soglie, sigma_dp = otsu_threshold_dp(gray, k)
        soglie_str = str(soglie)
        print(f"{name:<28} {t_our:>8d} {t_cv:>9d} {soglie_str:>20} {sigma_dp:>12.2f}")
    print("=" * 75)
    print(f"\nDP con k={k} soglie → {k+1} classi")
    print("Nota: con k=1 la soglia DP deve coincidere con t_otsu.\n")

if __name__ == "__main__":
    print("Caricamento immagini...")
    images = load_sample_images()
    print(f"  → {len(images)} immagini caricate: {list(images.keys())}\n")

    print_report(images, 5)
    plot_results(images, 5)