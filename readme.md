# Metodo Otsu

Riferimenti principali:
- A Threshold Selection Method for Gray-Level Histograms (M. Otsu - 1979) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4310076
- A Fast Algorithm for Multilevel Thresholding (Ping-Sung Liao, Tsu-Sheng Chen, Pau-Choo Chung - 2001) http://smile.ee.ncku.edu.tw/old/Links/MTable/ResearchPaper/papers/2001/A%20fast%20algorithm%20for%20multilevel%20%20thresholding.pdf
- Efficient Solution of Otsu Multilevel Image Thresholding: A Comparative Study (Mohamed H. Merzban, Mahmoud Elbayoumi - 2019) https://www.researchgate.net/publication/327455440_Efficient_Solution_of_Otsu_Multilevel_Image_Thresholding_A_Comparative_Study

# Estratto del Paper originale

Il **metodo Otsu** è un approccio di individuazione della soglia ottimale, all’interno di un’immagine digitale, per la suddivisione dei suoi livelli di grigio in due classi distinte. Questo metodo cerca di valutare la “bontà” della soglia per selezionarne automaticamente una ottimale.

Siano $L$ i livelli di grigio che possono assumere i pixel di un’immagine, $n_i$ il numero di pixel che assumono il livello di grigio $i$, $N=\sum_{n=1}^{L}n_i$ il numero totale dei pixel. Ora, prendiamo in considerazione l’istogramma normalizzato dell’immagine cioè la sua distribuzione di probabilità:

$$
P(i)=n_i/N
$$

$P(i)\ge0$ è la probabilità che un pixel sia del livello $i$

$\sum_{i=1}^{L}P(i)=1$

Supponendo di voler dividere i pixel tramite una **soglia** al livello $k$ ****otterremo due classi:

- $C_0$ contenente i pixel dei livelli $[1,\dots,k]$;
- $C_1$ quelli dei livelli $[k+1,\dots,L]$.

Possiamo quindi ricavare la probabilità cumulativa delle due classi:

$$
\omega_0=P(C_0)=\sum_{i=1}^tP(i)=\omega(t)\\\omega_1=P(C_1)=\sum_{i=t+1}^LP(i)=1-\omega(t)
$$

e i loro valori medi:

$$
\mu_0=\sum_{i=1}^tiP(i|C_0)=\sum_{i=1}^t\frac{iP(i)}{\omega_0}=\frac{\mu(t)}{\omega(t)}\\\mu_1=\sum_{i=t+1}^LiP(i|C_1)=\sum_{i=t+1}^L\frac{iP(i)}{\omega_1}=\frac{\mu_T-\mu(t)}{1-\omega(t)}
$$

considerando:

$$
\mu_T=\mu(L)\\\mu(t)=\sum_{i=1}^tiP(i)
$$

Consideriamo adesso la varianza delle classi, data da:

$$
\sigma_0^2=\sum_{i=1}^t(i-\mu_0)^2P(i|C_0)=\sum_{i=1}^t\frac{(i-\mu_0)^2P(i)}{\omega_0}\\\sigma_1^2=\sum_{i=t+1}^L(i-\mu_1)^2P(i|C_1)=\sum_{i=t+1}^L\frac{(i-\mu_1)^2P(i)}{\omega_1}
$$

e la seguente varianza **inter-classe**, cioè la somma pesata delle varianze delle classi sopra:

$$
\sigma_b^2=\sigma_T^2-[\omega_0\sigma_0^2+\omega_1\sigma_1^2]=\omega_0(\mu_0-\mu_T)^2+\omega_1(\mu_1-\mu_T)^2=\omega_0\omega_1(\mu_1-\mu_0)^2\\\text{dove }\sigma_T^2=\sum_{i=1}^L(i-\mu_T)^2P(i)
$$

La **soglia ottimale t*** sarà quindi ottenuta dalla massimizzazione della varianza inter-classe. In ottica implementativa, il calcolo di essa può essere svolto anche nella seguente maniera utilizzando le quantità cumulative:

$$
\sigma_B^2(t)=\frac{[\mu_T\omega(t)-\mu(t)]^2}{\omega(t)[1-\omega(t)]}
$$

E quindi l’ottimale sarà: 

$$
\sigma_b^2(t*)=\max_{1\le t<L}\sigma_b^2(t)
$$

# Implementazione in Python

## Thresholding semplice con una soglia (k=1)

```python
def otsu_threshold(gray: np.ndarray) -> int:
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

    return int(np.argmax(sigma_b_sq))           # l'indice in int della soglia ottimale
```

In questa implementazione la funzione *otsu_threshold* prende in input un array numpy contenente i vari pixel dell’immagine in bianco e nero di cui vogliamo ottenere la soglia ottimale che viene ritornata in output in *int*.

Per prima cosa andiamo a calcolare la distribuzione di probabilità normalizzata dell’immagine attraverso il metodo *compute_histogram* e inizializziamo l’array contente tutti i livelli della nostra immagine (8 bit, 2^8=256 livelli, 0-255).

Per il calcolo della massima varianza sfruttiamo ciò che abbiamo evidenziato prima cioè i valori cumulativi ottenibili a partire dalla distribuzione *hist* e dai livelli:

- *omega* è il nostro array delle probabilità della classe $C_0$ per ogni possibile soglia (indicata dall’indice nell’array);
- *mu_cum* è la media cumulativa calcolata come per *omega* ma moltiplicando la probabilità per il rispettivo livello;
- *mu_T* è per definizione la *mu_cum* dell’ultimo elemento cioè $\mu(L)$.

Calcoliamo quindi la varianza inter-classe *sigma_b_sq* come da definizione aggiungendo il vincolo 0 < *omega* < 1, evitando così i casi critici (oltre ad essere critico a livello matematico, per definizione andrebbe a suggerire che una delle due classi è vuota, cosa che non risulta utile nella ricerca della soglia ottimale) e i loro possibili warning.

Infine ritorniamo l’indice in *int* della massima varianza tra tutte quelle calcolate, che rappresenta la soglia ottimale.

## Multithresholding con DP (k>1)

Implementazione basato sull'approccio dettato da Mohamed H. Merzban e Mahmoud Elbayoumi nel paper *Efficient Solution of Otsu Multilevel Image Thresholding: A Comparative Study (2019)*.

L’evoluzione naturale del metodo Otsu è quella di suddividere l’immagine in più classi tramite la ricerca di un numero maggiore di soglie. Ad esempio con t=2 avrei 3 classi:

- $C_0$ contiene i livelli da $[1,\dots,t_1]$;
- $C_1$ contiene i livelli da $[t_1+1,...,t_2]$;
- $C_2$ contiene i livelli da $[t_2+1,...,L]$.

In generale con un numero di soglie pari a $k$ la funzione da ottimizzare è la seguente:

$$
\sigma_b^2(t_1^*,\dots,t_k^*)=\max_{1\le t_1<\dots<t_k<L}\sigma_b^2(t_1,\dots,t_k)
$$

Una generica classe definita dei boundaries [a, b] contribuisce alla varianza inter-classe come nel modo calcolato in questo metodo:

```python
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
```

```python
def otsu_threshold_dp(gray: np.ndarray, k: int) -> tuple[list[int], float]:
    
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
```

Come nel caso semplice vado a calcolare i dati cumulativi dell’immagine. Poi inizializzo una *variance_table* e un array parallelo *thresholds* che mi servono per mantenere la massima varianza tra le classi utilizzando un numeri *i* di soglie fino al livello *t* e il loro indice (quindi la soglia ottimale *t_k*).

Nel caso base dell’uso di 0 soglie la mia tabella delle varianze è pari al contributo della classe [0, t], mentre per le successive vado a calcolarlo nel seguente modo:

- affinché io possa usare *i* soglie ho bisogno di almeno *i+1* livelli, per cui l’upper bound *t* da considerare varia tra *i* e l’ultimo livello;
- a questo punto il mio range di possibili soglie *t_prev_range* è proprio da *i-1* fino al corrente *t*. Qui svolgo il calcolo della varianza, sommando al contributo della classe attuale (da *t_k + 1* fino a *t*) quello della classe (da 0 a *t_k*) che è proprio lo stesso calcolato all’iterazione di *i* precedente (cioè usando *i-1* soglie). Trovo il massimo e lo salvo nella *variance_table* e il suo indice in *thresholds;*
- infine con una sorta di “backtracking” costruisco la mia lista ottima di soglie selezionando ogni volta quella ottima affinché riesco a usare *i* soglie per i livelli da 0 a *t* (nel primo caso *t=255* perché vogliamo considerare tutta l’immagine). Il contenuto dell’array viene invertito.