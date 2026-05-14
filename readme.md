# Multi-Otsu Threshold — Benchmark su Mirflickr-25K

Confronto prestazionale tra un'implementazione DP dell'algoritmo di Otsu multi-soglia
e l'implementazione di riferimento di **scikit-image**, sul dataset Mirflickr-25K.

---

## Struttura del progetto

```
.
├── otsu.py                  # Implementazione proposta (otsu_threshold_dp)
├── bench_mirflickr25k.py    # Script di benchmark
├── bench_pyperf.py          # Benchmark con pyperf
├── requirements.txt
└── mirflickr_25k/           # Dataset (da scaricare separatamente, vedi sotto)
    └── mirflickr/
        └── *.jpg
```

---

## Requisiti

- Python 3.10+
- Il dataset Mirflickr-25K (vedi sotto)

---

## Setup

```bash
# 1. Clona il repository
git clone https://github.com/bpm-vb/MetodoOtsu.git
cd MetodoOtsu

# 2. Crea e attiva il virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Installa le dipendenze
pip install -r requirements.txt
```

---

## Dataset

Il dataset **Mirflickr-25K** non è incluso nel repository per via delle dimensioni.

1. Scaricalo da [https://press.liacs.nl/mirflickr/](https://press.liacs.nl/mirflickr/)
2. Estrai l'archivio nella cartella del progetto in modo che il path risultante sia:
   ```
   mirflickr_25k/mirflickr/*.jpg
   ```

---

## Utilizzo

```bash
python bench_mirflickr25k.py
```

I parametri principali si trovano in cima a `bench_mirflickr25k.py`:

| Parametro     | Default | Descrizione                                      |
|---------------|---------|--------------------------------------------------|
| `SEED`        | `42`    | Seed per la selezione riproducibile delle immagini |
| `WARMUP_IMGS` | `3`     | Immagini di warm-up escluse dalle statistiche    |
| `REPEATS`     | `3`     | Ripetizioni per immagine (si usa la mediana)     |
| `ATOL`        | `1`     | Tolleranza per il confronto delle soglie         |

Per modificare il numero di classi o la dimensione del dataset, recarsi nel main:

```python
CLASSES = 3
DATASET_DIM = 50
```

---

## Output

Al termine del benchmark vengono prodotti:

- **Statistiche a schermo** — media, mediana, deviazione standard, 95° percentile, tempo totale e speedup per entrambi gli algoritmi
- **Test di Wilcoxon** — verifica se la differenza di prestazioni è statisticamente significativa
- `results_otsu.npz` — array raw dei tempi, ricaricabile senza rieseguire il benchmark
- `results_otsu.csv` — una riga per run, utile per confrontare configurazioni diverse