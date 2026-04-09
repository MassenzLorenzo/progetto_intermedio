# 🧠 Student Mental Health Analysis
### Progetto Intermedio – Statistical Learning

> Analisi statistica dell'impatto delle abitudini quotidiane sul benessere psicologico degli studenti universitari.

---

## 📌 Descrizione

Questo progetto esplora la relazione tra **routine quotidiana** (ore di sonno, attività fisica) e il **benessere mentale** degli studenti, con particolare attenzione all'ansia pre-esame e alle differenze di genere.

Il dataset utilizzato è **"Student Mental Health Analysis"** (fonte: Kaggle), composto da **1.000 osservazioni reali** di studenti universitari.

---

## ❓ Domande di Ricerca

| # | Domanda Statistica | Metodo |
|---|---|---|
| 1 | Esistono differenze significative nelle **ore di sonno** tra chi prova ansia pre-esame e chi no? | Test t per campioni indipendenti |
| 2 | C'è una **correlazione positiva** tra ore di attività fisica e ore di sonno? | Correlazione di Pearson |
| 3 | C'è un'**associazione significativa** tra Genere e ansia pre-esame? | Test Chi-Quadro |

---

## 📂 Struttura del Progetto

```
progetto_intermedio/
│
├── csv/
│   └── Student_Mental_Health.csv       # Dataset originale (1.000 studenti)
│
├── student_mental_health.py            # Script Python principale
├── Progetto Statistical Learning.pdf   # Relazione completa del progetto
├── student_mental_health.docx          # Documento di supporto
└── README.md
```

---

## 🔢 Variabili Chiave

| Variabile | Tipo | Descrizione |
|---|---|---|
| `sleep_hours` | Quantitativa continua | Ore di sonno notturno |
| `sport_hours` | Quantitativa continua | Ore di attività fisica settimanale |
| `anxiety_status` | Categoriale dicotomica | Ansia prima degli esami (`Yes` / `No`) |
| `gender` | Categoriale nominale | Genere dello studente |

---

## 🔬 Metodologia

### 1. Data Understanding & Preprocessing
- Caricamento e ispezione del dataset
- Analisi dei valori mancanti (risultato: **0 missing values**)
- Rinomina delle colonne per chiarezza
- Verifica outlier tramite **boxplot** → nessun outlier rilevato

### 2. Statistica Descrittiva
- Calcolo di media, mediana, deviazione standard e varianza per le variabili quantitative
- Distribuzione delle frequenze per le variabili categoriali
- Visualizzazioni: **istogramma**, **boxplot per gruppo**, **scatter plot**

### 3. Analisi di Correlazione
- **Correlazione di Pearson** tra `sport_hours` e `sleep_hours`
- Risultato: `r = -0.0134`, `p-value = 0.6720` → **nessuna correlazione significativa**

### 4. Analisi Inferenziale

#### Test 1 – Test t per campioni indipendenti (Sonno vs Ansia)
```
H₀: Non c'è differenza nelle ore di sonno tra i due gruppi
H₁: C'è una differenza significativa
α = 0.05
```
→ **p-value > 0.05**: Non si rifiuta H₀. Le ore di sonno non differiscono significativamente in base allo stato d'ansia.

#### Test 2 – Test Chi-Quadro (Genere vs Ansia)
```
H₀: Non c'è associazione tra Genere e Ansia Pre-Esame
H₁: Esiste un'associazione significativa
α = 0.05
```
→ **p-value > 0.05**: Non si rifiuta H₀. Il genere non è associato significativamente all'ansia pre-esame.

---

## 📊 Risultati Principali

- ✅ Le abitudini di sonno degli studenti sono **distribuite in modo quasi uniforme** (nessun valore centrale dominante).
- ✅ L'ansia pre-esame **non influenza significativamente** la quantità di sonno notturno.
- ✅ Non esiste una **correlazione** tra attività fisica e ore di sonno nel campione analizzato.
- ✅ Il **genere** non risulta un fattore determinante per la presenza di ansia pre-esame.

> ⚠️ **Nota importante**: correlazione ≠ causalità. L'assenza di legami statistici significativi nel campione non esclude possibili relazioni più complesse non lineare o confondenti.

---

## 🛠️ Tecnologie Utilizzate

| Libreria | Utilizzo |
|---|---|
| `pandas` | Caricamento e manipolazione del dataset |
| `numpy` | Operazioni numeriche |
| `matplotlib` | Creazione dei grafici |
| `seaborn` | Visualizzazioni statistiche avanzate |
| `scipy.stats` | Test statistici (t-test, Chi-Quadro, Pearson) |

---

## ▶️ Come Eseguire

1. **Clona il repository**
   ```bash
   git clone https://github.com/<tuo-username>/progetto_intermedio.git
   cd progetto_intermedio
   ```

2. **Installa le dipendenze**
   ```bash
   pip install pandas numpy matplotlib seaborn scipy
   ```

3. **Esegui lo script**
   ```bash
   python student_mental_health.py
   ```

---

## 👤 Autore

**Lorenzo Massenz**  
Corso di Statistical Learning – Progetto Intermedio  
Anno Accademico 2025/2026

---

## 📄 Fonte Dataset

Dataset: [Student Mental Health Analysis – Kaggle](https://www.kaggle.com/)  
Osservazioni: 1.000 studenti reali
