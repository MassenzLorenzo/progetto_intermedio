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