# Import niezbędnych bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Wczytanie danych z pliku CSV
file_path = "C:/Users/MSI/Desktop/10.1/titanic.csv"
dataset = pd.read_csv(file_path)

# Usunięcie wartości brakujących w kolumnie Fare
fare_values = dataset['Fare'].dropna()
print("Liczba obserwacji w kolumnie Fare:", len(fare_values))

# Obliczenie statystyk
mean_fare = np.round(np.mean(fare_values), 2)
median_fare = np.round(np.median(fare_values), 2)
mode_result = stats.mode(fare_values, keepdims=True)
mode_fare = np.round(mode_result.mode[0], 2)

print("Średnia:", mean_fare)
print("Mediana:", median_fare)
print("Dominanta (moda):", mode_fare)

# Normalizacja danych
scaler_standard = StandardScaler()
fare_scaled_standard = scaler_standard.fit_transform(fare_values.values.reshape(-1, 1))

scaler_minmax = MinMaxScaler()
fare_scaled_minmax = scaler_minmax.fit_transform(fare_values.values.reshape(-1, 1))

# Histogram – standaryzacja
plt.figure()
plt.hist(fare_scaled_standard, bins=30)
plt.title("Standaryzowana kolumna Fare")
plt.xlabel("Fare (standaryzowana)")
plt.ylabel("Liczba pasażerów")
plt.grid(True)
plt.show()

# Histogram – MinMax
plt.figure()
plt.hist(fare_scaled_minmax, bins=30)
plt.title("Znormalizowana (Min-Max) kolumna Fare")
plt.xlabel("Fare (Min-Max)")
plt.ylabel("Liczba pasażerów")
plt.grid(True)
plt.show()

# WYKRES PUDEŁKOWY – PEŁNE DANE (może być nieczytelny przez outliery)
plt.figure()
plt.boxplot(fare_values, vert=False, patch_artist=True)
plt.title("Wykres pudełkowy ceny biletu (pełne dane)")
plt.xlabel("Cena biletu")
plt.grid(True)
plt.show()

# WYKRES PUDEŁKOWY – Z OGRANICZONĄ OSIĄ X (lepsza czytelność)
plt.figure()
plt.boxplot(fare_values, vert=False, patch_artist=True)
plt.title("Wykres pudełkowy ceny biletu (ograniczenie do 100)")
plt.xlabel("Cena biletu")
plt.xlim(0, 100)  # Zakres X ograniczony do 100 jednostek
plt.grid(True)
plt.show()

# WYKRES PUDEŁKOWY – BEZ OUTLIERÓW (>99 centyla)
filtered_fare = fare_values[fare_values < fare_values.quantile(0.99)]

plt.figure()
plt.boxplot(filtered_fare, vert=False, patch_artist=True)
plt.title("Wykres pudełkowy ceny biletu (bez wartości odstających >99 centyla)")
plt.xlabel("Cena biletu")
plt.grid(True)
plt.show()
