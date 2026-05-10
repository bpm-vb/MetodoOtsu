import numpy as np


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
    Calcola la soglia ottimale seguendo l'algoritmo di Otsu (1979).

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
    hist = compute_histogram(gray, normalize=True)   # P(i)
    levels = np.arange(256, dtype=np.float64)

    # Calcolo dei valori cumulativi
    omega = np.cumsum(hist)                     # P(C0) per le possibili soglie
    mu_cum = np.cumsum(hist * levels)           # Media cumulativa per le possibili soglie
    mu_T = mu_cum[-1]                           # Media globale

    # Varianza inter-classe per tutte le possibili soglie in una sola riga vettorizzata
    with np.errstate(divide='ignore', invalid='ignore'):
        sigma_b_sq = np.where(
            (omega > 0) & (omega < 1),
            (mu_T * omega - mu_cum) ** 2 / (omega * (1.0 - omega)),
            0.0
        )

    idx = np.argmax(sigma_b_sq)

    return int(idx), float(sigma_b_sq[idx])     # l'indice in int della soglia ottimale

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

    return omega_ab * (mu_ab - mu_T)**2

def otsu_threshold_dp(gray: np.ndarray, k: int) -> tuple[list[int], float]:
    """
    Calcola le soglie ottimali seguendo l'approccio di 
    Mohamed H. Merzban e Mahmoud Elbayoumi (2019)
    """

    # Istogramma e cumulativi
    hist = compute_histogram(gray, normalize=True)   # p(i)
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(hist)                          # P(C0) per le possibili soglie
    mu_cum = np.cumsum(hist * levels)                # Media cumulativa per le possibili soglie
    mu_T = mu_cum[-1]                                # Media globale

    # Tabella con soglie e livelli
    variance_table = np.zeros((k, 256))   # max varianza tra classi usando i soglie fino al livello t
    thresholds = np.zeros((k, 256))       # memorizza la soglia t_k che ha generato il massimo

    for t in range(256): # caso base in cui utilizzo 0 soglie
        variance_table[0][t] = class_contribution(0, t, omega, mu_cum, mu_T)

    for i in range(1, k): # utilizzo le soglie fino a k
        for t in range(i, 256): # itero sui livelli, da 0 a [i, 256], parto da i perché così posso calcolare le mie k=i soglie altrimenti non ci riuscirei avendo un array più piccolo
            t_prev_range = np.arange(i-1, t) # range di possibili soglie

            # Calcolo i contributi della classe per ogni soglie precedente
            candidates = [
                variance_table[i-1][t_k] + class_contribution(t_k+1, t, omega, mu_cum, mu_T)
                for t_k in t_prev_range
            ]
            
            if candidates:
                best_idx = np.argmax(candidates)
                variance_table[i][t] = candidates[best_idx]
                thresholds[i][t] = t_prev_range[best_idx]

    best_thresholds = []
    t = 255
    for i in range(k-1, 0, -1):
        t = int(thresholds[i][t])
        best_thresholds.append(t)
    best_thresholds.reverse()

    return best_thresholds, variance_table[k-1][255]

def otsu_threshold_backtracking(gray: np.ndarray, k: int):
    # inizializziamo le variabili per i valori cumulativi
    hist = compute_histogram(gray, normalize=True)
    levels = np.arange(256)
    omega = np.zeros(257)
    mu_cum = np.zeros(257)
    omega[1:] = np.cumsum(hist)
    mu_cum[1:] = np.cumsum(hist * levels)
    mu_T = mu_cum[-1]

    vbest = -1      # migliore varianza
    tbest = []      # migliore configurazione (lista di soglie)
    
    memo_cost = {}  # dizionario per il contributo della classe [a,b]

    def get_cost(a, b):     # class contribuition ottimizzata
        if (a, b) in memo_cost:
            return memo_cost[(a, b)]
        omega_ab = omega[b + 1] - omega[a]
        if omega_ab <= 0:
            return 0.0
        mu_ab = (mu_cum[b + 1] - mu_cum[a]) / omega_ab
        sigma_b_sq_ab = omega_ab * (mu_ab - mu_T)**2
        memo_cost[(a,b)] = sigma_b_sq_ab
        return sigma_b_sq_ab
    
    def backtrack(start_level, k, vcurr, tcurr):
        nonlocal vbest, tbest

        if k == 0:  # caso base con 0 soglie da piazzare
            variance = vcurr + get_cost(start_level, 255)
            if variance > vbest:
                vbest = variance
                tbest = list(tcurr)
            return
        
        # piazziamo la prossima soglia
        for s in range(start_level, 255 - k):
            variance_step = get_cost(start_level, s)
            tcurr.append(s)
            backtrack(s + 1, k - 1, vcurr + variance_step, tcurr)
            tcurr.pop()
    
    backtrack(0, k, 0.0, [])

    return tbest, vbest