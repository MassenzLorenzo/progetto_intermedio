"""
PROGETTO INTERMEDIO - STATISTICAL LEARNING
Analisi della Salute Mentale e delle Abitudini degli Studenti

Dataset: "Student Mental Health Analysis" (Fonte: Kaggle)
Osservazioni: 1.000 studenti reali.

Obiettivo: 
Indagare l'impatto della routine quotidiana (sonno, sport) sul benessere 
psicologico degli studenti e verificare eventuali differenze demografiche.

Domande Statistiche:
1. Esistono differenze significative nelle ore di sonno tra chi prova ansia 
   pre-esame e chi non la prova? (Test inferenziale su differenze tra gruppi)
2. C'è una correlazione positiva tra le ore di attività fisica e le ore di sonno? (Correlazione)
3. C'è un'associazione significativa tra il Genere e l'ansia pre-esame? (Test Chi-Quadro)

Variabili Chiave Utilizzate:
- Sleep Duration (Quantitativa continua): Ore di sonno notturno.
- Physical Activity (Quantitativa continua): Ore di sport a settimana.
- Anxious Before Exams (Categoriale dicotomica): Ansia prima degli esami (Yes/No).
- Gender (Categoriale nominale): Genere dello studente.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# ==========================================
# 4.2 DATA UNDERSTANDING E PREPROCESSING
# ==========================================

# 1. Caricamento del dataset
df = pd.read_csv('csv/Student_Mental_Health.csv')

# Visualizziamo le prime righe
print("--- Prime 5 righe del dataset ---")
print(df.head())

# Verifichiamo le informazioni generali
print("\n--- Informazioni sul dataset ---")
print(df.info())

# 2. Analisi dei valori mancanti (Missing Values)
print("\n--- Conteggio dei valori mancanti per colonna ---")
print(df.isnull().sum())

""" 
Valori mancanti: 0, quindi non è necessario alcun trattamento per i missing values.
comando per listwise deletion: df = df.dropna(subset=colonne_chiave)
"""

# Rinominiamo le colonne per una maggiore chiarezza e coerenza con le domande statistiche
df = df.rename(columns={
    'Sleep Duration (hrs)': 'sleep_hours',
    'Physical Activity (hrs/week)': 'sport_hours',
    'Anxious Before Exams': 'anxiety_status',
    'Gender': 'gender'
})

# Selezioniamo solo le colonne che ci servono per le nostre domande
colonne_chiave = ['sleep_hours', 'sport_hours', 'anxiety_status', 'gender']

# 3. Gestione Outlier (Valori anomali)
# Visualizziamo i boxplot per identificare visivamente eventuali anomalie
plt.figure(figsize=(12, 6))

# utilizziamo subplot per disporre i grafici affiancati 
plt.subplot(1, 2, 1) # righe, colonne, indice del grafico
sns.boxplot(y=df['sleep_hours'], color='skyblue')
plt.title('Distribuzione Ore di Sonno')
plt.ylabel('Ore')

plt.subplot(1, 2, 2) # righe, colonne, indice del grafico
sns.boxplot(y=df['sport_hours'], color='lightgreen')
plt.title('Distribuzione Attività Fisica (ore/settimana)')
plt.ylabel('Ore')

plt.tight_layout() 
plt.show()

"""
Attraverso l'ispezione visiva dei boxplot per le variabili sleep_hours e sport_hours,
si osserva la totale assenza di outlier (rappresentati graficamente da punti isolati oltre i baffi).
Questo indica che il dataset è estremamente pulito e che le abitudini dichiarate dagli studenti
sono verosimili e prive di anomalie statistiche.
Di conseguenza, non è stato necessario procedere alla rimozione di alcuna osservazione, 
mantenendo l'integrità dell'intero campione originale per le analisi successive.
"""

# ==========================================
# 4.3 STATISTICA DESCRITTIVA E VISUALIZZAZIONI
# ==========================================

print("\n--- 4.3 STATISTICA DESCRITTIVA ---")

# 1. Misure di posizione e dispersione per le variabili quantitative
print("\nStatistiche Ore di Sonno:")
print(f"Media: {df['sleep_hours'].mean():.2f}")
print(f"Mediana: {df['sleep_hours'].median():.2f}")
print(f"Deviazione Standard: {df['sleep_hours'].std():.2f}")
print(f"Varianza: {df['sleep_hours'].var():.2f}")

print("\nStatistiche Ore di Sport:")
print(f"Media: {df['sport_hours'].mean():.2f}")
print(f"Mediana: {df['sport_hours'].median():.2f}")
print(f"Deviazione Standard: {df['sport_hours'].std():.2f}")
print(f"Varianza: {df['sport_hours'].var():.2f}")

# 2. Frequenze per le variabili categoriali
print("\nConteggio Genere:")
print(df['gender'].value_counts())
print("\nConteggio Ansia Pre-Esame:")
print(df['anxiety_status'].value_counts())

# 3. Visualizzazioni
# Creiamo una figura con 3 grafici affiancati (1 riga, 3 colonne)
plt.figure(figsize=(18, 5))

# Grafico 1: Istogramma (Distribuzione delle ore di sonno)
plt.subplot(1, 3, 1)
# kde -> curva di densità (la usiamo per visualizzare meglio la distribuzione)
sns.histplot(df['sleep_hours'], kde=True, bins=15, color='royalblue')
plt.title('Distribuzione Ore di Sonno')
plt.xlabel('Ore di sonno')
plt.ylabel('Frequenza')
""" 
possiamo vedere come la distribuzione delle ore di sonno sia quasi uniforme,
indicando che le abitudini di riposo degli studenti non si concentrano su un valore medio universale.
"""
# Grafico 2: Boxplot (Confronto Sonno per gruppo di Ansia)
plt.subplot(1, 3, 2)
sns.boxplot(data=df, x='anxiety_status', y='sleep_hours', palette='Set2')
plt.title('Ore di Sonno per Stato d\'Ansia')
plt.xlabel('Ansia Pre-Esame')
plt.ylabel('Ore di Sonno')
"""
il boxplot non mostra differenze evidenti tra i due gruppi, con mediane e distribuzioni simili.
Questo suggerisce che potrebbe non esserci una relazione chiara tra l'ansia pre-esame e le ore di sonno.
"""
# Grafico 3: Scatter plot (Relazione tra Sport e Sonno)
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='sport_hours', y='sleep_hours', alpha=0.6, color='darkgreen')
plt.title('Relazione tra Sport e Sonno')
plt.xlabel('Ore di sport (settimanali)')
plt.ylabel('Ore di sonno (notturne)')
"""
lo scatter plot mostra un'assenza totale di relazione lineare o monotona tra 
l'attività fisica settimanale e la durata del sonno notturno.
"""
# Mostriamo i grafici in modo ordinato
plt.tight_layout()
plt.show()

# ==========================================
# 4.4 ANALISI DI CORRELAZIONE
# ==========================================

# DOMANDA: c'è una correlazione positiva tra le ore di attività fisica e le ore di sonno?

""" Avendo già verificato l'assenza di outlier, useremo la correlazione di Pearson, 
 adatta per variabili quantitative continue senza anomalie.

 La correlazione di Pearson misura la forza e la direzione di una relazione lineare tra due variabili.
 Il coefficiente di correlazione (r) varia tra -1 e 1, dove:
    - r = 1 indica una correlazione positiva perfetta (le variabili aumentano insieme)
    - r = -1 indica una correlazione negativa perfetta (una variabile aumenta mentre l'altra diminuisce)
    - r = 0 indica nessuna correlazione lineare (le variabili non hanno relazione lineare)
"""
from scipy.stats import pearsonr

# Estraiamo le variabili di interesse
sport = df['sport_hours']
sonno = df['sleep_hours']

# Calcoliamo il coefficiente di correlazione di Pearson e il p-value
r, p_value = pearsonr(sport, sonno)

print("\nAnalisi di Correlazione tra Ore di Sport e Ore di Sonno:")
print(f"\nCoefficiente di Correlazione (r): {r:.4f}")
print(f"\nP-value: {p_value:.4f}")

"""
Il calcolo ha restituito un coefficiente r pari a -0.0134 con un p-value di 0.6720. 
Essendo il coefficiente vicinissimo allo zero, deduciamo un'assenza totale di correlazione lineare. 
La direzione (negativa) risulta ininfluente data la debolezza del legame.
"""

"""
Coerenza con il contesto: "Questo risultato è coerente con quanto osservato visivamente nello scatter plot precedente.
Nel contesto degli studenti analizzati, l'ammontare di attività fisica settimanale non sembra 
essere un fattore che influenza in alcun modo la durata del sonno notturno.
"""

# ==========================================
# 4.5 ANALISI STATISTICA INFERENZIALE
# ==========================================

""" Fissiamo il livello di significatività standard (alpha = 0.05) per tutti i test inferenziali, 
che rappresenta la soglia oltre la quale rifiuteremo l'ipotesi nulla (H0).
"""
alpha = 0.05

# ---------------------------------------------------------
# TEST 1: Test t per campioni indipendenti (Sonno vs Ansia)
# ---------------------------------------------------------

"""
il test t per campioni indipendenti è adatto per confrontare le medie di due gruppi distinti 
(in questo caso, studenti con ansia pre-esame vs studenti senza ansia) sulla variabile quantitativa continua "ore di sonno".

H0 (Ipotesi Nulla): "Non c'è differenza nella media delle ore di sonno tra chi prova ansia pre-esame e chi non la prova."
H1 (Ipotesi Alternativa): "C'è una differenza significativa nella media delle ore di sonno tra i due gruppi."
"""

# Separiamo le ore di sonno nei due gruppi basati sull'ansia
sleep_anxious = df[df['anxiety_status'] == 'Yes']['sleep_hours']
sleep_not_anxious = df[df['anxiety_status'] == 'No']['sleep_hours']

# Eseguiamo il test t (equal_var=False è più robusto se le varianze sono diverse)
t_stat, p_val_t = stats.ttest_ind(sleep_anxious, sleep_not_anxious, equal_var=False)

# Stampiamo i risultati del test t -> statistica t e p-value
print("\n--- TEST 1: Test t per campioni indipendenti (Sonno vs Ansia) ---")
print(f"\nStatistica t: {t_stat:.3f}")
print(f"\nP-value: {p_val_t:.3e}")

if p_val_t < alpha:
    print("Risultato: Rifiutiamo l'ipotesi nulla (H0). C'è una differenza significativa.")
else:
    print("Risultato: Non possiamo rifiutare l'ipotesi nulla (H0). Non c'è una differenza significativa.")

""" 
Essendo il p-value maggiore di 0.05, non rifiutiamo l'ipotesi nulla.
Coerenza con il contesto: "Il test t ha confermato quanto osservato visivamente nel boxplot,
indipendentemente dallo stato d'ansia pre-esame, gli studenti tendono a dormire un numero simile di ore.
Questo suggerisce che l'ansia pre-esame non è un fattore che influenza significativamente la quantità di sonno notturno tra gli studenti analizzati.
"""


# ---------------------------------------------------------
# TEST 2: Test Chi-Quadro (Genere vs Ansia)
# ---------------------------------------------------------
"""
Il test Chi-Quadro di indipendenza è adatto per verificare se esiste un'associazione
significativa tra due variabili categoriali (in questo caso, Genere e Ansia Pre-Esame).
H0 (Ipotesi Nulla): "Non c'è associazione tra Genere e Ansia Pre-Esame. Le variabili sono indipendenti."
H1 (Ipotesi Alternativa): "C'è un'associazione significativa tra Genere e Ansia Pre-Esame. Le variabili non sono indipendenti."
"""

print("\n--- TEST 2: Test Chi-Quadro (Genere vs Ansia) ---")

# Creiamo la tabella di contingenza (frequenze incrociate)
contingency_table = pd.crosstab(df['gender'], df['anxiety_status'])
print("Tabella di contingenza:")
print(contingency_table)

# Eseguiamo il test Chi-Quadro
chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency_table) 
"""restituisce la statistica chi2 (che misura l'associazione tra le variabili), 
il p-value, i gradi di libertà (che spiegano la distribuzione del chi-quadro) 
e le frequenze attese (che rappresentano la distribuzione teorica se le variabili fossero indipendenti)
"""
print(f"\nStatistica Chi-Quadro: {chi2_stat:.3f}")
print(f"\nGradi di libertà (dof): {dof}")
print(f"\nP-value: {p_val_chi2:.3e}")

if p_val_chi2 < alpha:
    print("Risultato: Rifiutiamo l'ipotesi nulla (H0). C'è un'associazione significativa.")
else:
    print("Risultato: Non possiamo rifiutare l'ipotesi nulla (H0). Le variabili sono indipendenti.")

"""
Il test Chi-Quadro ha restituito un p-value maggiore di 0.05, quindi non rifiutiamo l'ipotesi nulla.
Coerenza con il contesto: "Il test ha confermato quanto osservato nella tabella di contingenza,
indipendentemente dal genere, gli studenti mostrano una distribuzione simile di ansia pre-esame.
Questo suggerisce che il genere non è un fattore che influenza significativamente la presenza di ansia pre-esame tra gli studenti analizzati.
"""

