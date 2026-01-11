import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
X = iris.data[:, :2]   # 2 pierwsze cechy

print('Dane załadowane:')
print(f'Liczba próbek: {X.shape[0]}')
print(f'Cechy {iris.feature_names[:2]}')

# 1. Metoda łokcia - znajdowanie optymalnej liczby klastrów
inercje = []
liczby_klastrow = range(1, 9)
for k in liczby_klastrow:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inercje.append(kmeans.inertia_)
    print(f'K={k}: inercja = {kmeans.inertia_:.2f}')

# Wykres metody łokcia
plt.plot(liczby_klastrow, inercje, 'bo-', linewidth=2, markersize=8)
plt.title('Metoda łokcia')
plt.xlabel('Liczba klastrów (K)')
plt.ylabel('Inercje')
plt.grid(True, alpha=0.3)
plt.show()