# Import potrzebnych bibliotek
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Funkcja do obliczania statystyk opisowych i ich wyświetlania
def describe_stats(data, name):
    print(f"Statystyki dla {name}:")
    print(f"Liczba obserwacji: {len(data)}")
    print(f"Średnia: {np.mean(data):.4f}")
    print(f"Mediana: {np.median(data):.4f}")
    mode_result = stats.mode(data, keepdims=True)
    print(f"Moda: {mode_result.mode[0]:.4f} (wystąpień: {mode_result.count[0]})")
    print(f"Min: {np.min(data):.4f}")
    print(f"Max: {np.max(data):.4f}")
    print(f"Q1 (25%): {np.quantile(data, 0.25):.4f}")
    print(f"Q3 (75%): {np.quantile(data, 0.75):.4f}")
    print(f"Wariancja: {np.var(data, ddof=1):.4f}")
    print(f"Odchylenie standardowe: {np.std(data, ddof=1):.4f}")
    print(f"Asymetria: {stats.skew(data):.4f}")
    print(f"Kurtosis: {stats.kurtosis(data):.4f}")
    print("-"*40)

# Ustawienia parametrów rozkładu normalnego
mean = 3
std_dev = 1

# Generowanie danych
np.random.seed(42)  # dla powtarzalności wyników
data_100 = np.random.normal(loc=mean, scale=std_dev, size=100)
data_10000 = np.random.normal(loc=mean, scale=std_dev, size=10000)

# Wyświetlanie statystyk dla obu zbiorów
describe_stats(data_100, "Rozkład normalny - 100 próbek")
describe_stats(data_10000, "Rozkład normalny - 10 000 próbek")

# Wykresy histogramów dla wizualizacji rozkładu danych
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(data_100, bins=20, color='skyblue', edgecolor='black')
plt.title("Histogram: 100 próbek")
plt.xlabel("Wartość")
plt.ylabel("Częstość")

plt.subplot(1,2,2)
plt.hist(data_10000, bins=20, color='salmon', edgecolor='black')
plt.title("Histogram: 10 000 próbek")
plt.xlabel("Wartość")
plt.ylabel("Częstość")

plt.tight_layout()
plt.show()