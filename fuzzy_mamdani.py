import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# === 1. Load dataset ===
df = pd.read_csv("student_sleep_patterns.csv")

# === 2. Normalisasi data (hanya kolom yang dipakai) ===
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df[["Sleep_Duration", "Physical_Activity", "Caffeine_Intake", "Sleep_Quality"]])
df_norm = pd.DataFrame(normalized_data, columns=["Sleep_Duration", "Physical_Activity", "Caffeine_Intake", "Sleep_Quality"])

print("Data setelah normalisasi:")
print(df_norm.head())

# === 3. Definisi variabel fuzzy sesuai soal ===
sleep_duration = ctrl.Antecedent(np.arange(4, 9.1, 0.1), 'Sleep_Duration')
activity       = ctrl.Antecedent(np.arange(0, 120.1, 1), 'Physical_Activity')
caffeine       = ctrl.Antecedent(np.arange(0, 5.1, 0.1), 'Caffeine_Intake')
sleep_quality  = ctrl.Consequent(np.arange(1, 10.1, 0.1), 'Sleep_Quality')

# === 4. Membership functions ===
# Durasi Tidur
sleep_duration['pendek']  = fuzz.trapmf(sleep_duration.universe, [4, 4, 6.5, 6.5])
sleep_duration['sedang']  = fuzz.trimf(sleep_duration.universe, [5, 6.5, 8])
sleep_duration['panjang'] = fuzz.trapmf(sleep_duration.universe, [6.5, 9, 9, 9])

# Aktivitas Fisik
activity['rendah'] = fuzz.trapmf(activity.universe, [0, 0, 60, 60])
activity['sedang'] = fuzz.trimf(activity.universe, [30, 60, 90])
activity['tinggi'] = fuzz.trapmf(activity.universe, [60, 120, 120, 120])

# Kafein
caffeine['rendah'] = fuzz.trapmf(caffeine.universe, [0, 0, 2.5, 2.5])
caffeine['sedang'] = fuzz.trimf(caffeine.universe, [1, 2.5, 4])
caffeine['tinggi'] = fuzz.trapmf(caffeine.universe, [3, 5, 5, 5])

# Kualitas Tidur
sleep_quality['buruk']       = fuzz.trapmf(sleep_quality.universe, [1, 1, 5, 5])
sleep_quality['cukup']       = fuzz.trimf(sleep_quality.universe, [3, 5, 7])
sleep_quality['baik']        = fuzz.trimf(sleep_quality.universe, [5, 10, 10])
sleep_quality['sangat_baik'] = fuzz.trimf(sleep_quality.universe, [7, 9, 10])

# === 5. Rule Base (24 aturan) ===
rules = [
    ctrl.Rule(sleep_duration['pendek'] & activity['rendah'] & caffeine['tinggi'], sleep_quality['buruk']),
    ctrl.Rule(sleep_duration['pendek'] & activity['rendah'] & caffeine['sedang'], sleep_quality['buruk']),
    ctrl.Rule(sleep_duration['pendek'] & activity['rendah'] & caffeine['rendah'], sleep_quality['buruk']),
    ctrl.Rule(sleep_duration['pendek'] & activity['sedang'] & caffeine['tinggi'], sleep_quality['buruk']),
    ctrl.Rule(sleep_duration['pendek'] & activity['sedang'] & caffeine['sedang'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['pendek'] & activity['sedang'] & caffeine['rendah'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['pendek'] & activity['tinggi'] & caffeine['tinggi'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['pendek'] & activity['tinggi'] & caffeine['sedang'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['pendek'] & activity['tinggi'] & caffeine['rendah'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['sedang'] & activity['rendah'] & caffeine['tinggi'], sleep_quality['buruk']),
    ctrl.Rule(sleep_duration['sedang'] & activity['rendah'] & caffeine['sedang'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['sedang'] & activity['rendah'] & caffeine['rendah'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['sedang'] & activity['sedang'] & caffeine['tinggi'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['sedang'] & activity['sedang'] & caffeine['sedang'], sleep_quality['baik']),
    ctrl.Rule(sleep_duration['sedang'] & activity['sedang'] & caffeine['rendah'], sleep_quality['baik']),
    ctrl.Rule(sleep_duration['sedang'] & activity['tinggi'] & caffeine['tinggi'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['sedang'] & activity['tinggi'] & caffeine['sedang'], sleep_quality['baik']),
    ctrl.Rule(sleep_duration['sedang'] & activity['tinggi'] & caffeine['rendah'], sleep_quality['sangat_baik']),
    ctrl.Rule(sleep_duration['panjang'] & activity['rendah'] & caffeine['tinggi'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['panjang'] & activity['rendah'] & caffeine['sedang'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['panjang'] & activity['rendah'] & caffeine['rendah'], sleep_quality['baik']),
    ctrl.Rule(sleep_duration['panjang'] & activity['sedang'] & caffeine['tinggi'], sleep_quality['cukup']),
    ctrl.Rule(sleep_duration['panjang'] & activity['sedang'] & caffeine['sedang'], sleep_quality['baik']),
    ctrl.Rule(sleep_duration['panjang'] & activity['sedang'] & caffeine['rendah'], sleep_quality['sangat_baik']),
]

# === 6. Sistem kontrol Mamdani ===
sleep_ctrl = ctrl.ControlSystem(rules)
sleep_sim = ctrl.ControlSystemSimulation(sleep_ctrl)

# === 7. Simulasi 24 percobaan ===
print("\n=== 24 Percobaan Prediksi Sleep Quality ===")
results = []
for i in range(24):
    sample = df.iloc[i]
    sleep_sim.input['Sleep_Duration'] = sample['Sleep_Duration']
    sleep_sim.input['Physical_Activity'] = sample['Physical_Activity']
    sleep_sim.input['Caffeine_Intake'] = sample['Caffeine_Intake']
    
    sleep_sim.compute()
    hasil = sleep_sim.output['Sleep_Quality']
    results.append(hasil)
    print(f"Percobaan {i+1}: Durasi={sample['Sleep_Duration']} jam, Aktivitas={sample['Physical_Activity']}, Kafein={sample['Caffeine_Intake']} → Prediksi Kualitas Tidur = {hasil:.2f}")

# === 8. Visualisasi 24 percobaan dalam 1 gambar grid ===
fig, axes = plt.subplots(6, 4, figsize=(24, 18))  # 6 baris × 4 kolom = 24 grafik
axes = axes.flatten()

for i in range(24):
    sample = df.iloc[i]
    sleep_sim.input['Sleep_Duration'] = sample['Sleep_Duration']
    sleep_sim.input['Physical_Activity'] = sample['Physical_Activity']
    sleep_sim.input['Caffeine_Intake'] = sample['Caffeine_Intake']
    sleep_sim.compute()
    
    sleep_quality.view(sim=sleep_sim, ax=axes[i])
    axes[i].set_title(f"Percobaan {i+1}")

plt.tight_layout()
plt.show()
