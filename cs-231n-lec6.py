# Gerekli kütüphaneler
import numpy as np
import matplotlib.pyplot as plt

# 1. Gaussian dağılımdan gelen 1000 örnekli 500 boyutlu input verisi oluştur
D = np.random.randn(1000, 500)

# 2. 10 katmanlı, her biri 500 nöronlu bir yapay sinir ağı belirle
hidden_layer_sizes = [500] * 10
nonlinearities = ['tanh'] * len(hidden_layer_sizes)

# 3. Aktivasyon fonksiyonlarını tanımla
act = {
    'relu': lambda x: np.maximum(0, x),
    'tanh': lambda x: np.tanh(x)
}

# 4. Katman çıktılarının tutulacağı dictionary
Hs = {}

# 5. Katmanları sırayla işle
for i in range(len(hidden_layer_sizes)):
    x = D if i == 0 else Hs[i - 1]  # ilk katmanda giriş verisi D, sonrası bir önceki katman
    fan_in = x.shape[1]
    fan_out = hidden_layer_sizes[i]
    
    # Ağırlıkları küçük random değerlerle başlat (0.01 ile çarpılmış)
    W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in/ 2) 
    # Matrix çarpımı + aktivasyon
    H = np.dot(x, W)
    H = act[nonlinearities[i]](H)
    
    # Hesaplanan çıktıyı kaydet
    Hs[i] = H

# 6. İlk input katmanının ortalaması ve standart sapması
print('Input layer mean: {:.4f}, std: {:.4f}'.format(np.mean(D), np.std(D)))

# 7. Her katman için ortalama ve std hesapla
layer_means = [np.mean(H) for H in Hs.values()]
layer_stds = [np.std(H) for H in Hs.values()]

# 8. Sonuçları yazdır
for i in range(len(hidden_layer_sizes)):
    print('Hidden layer {:2d} -> mean: {:.4f}, std: {:.4f}'.format(i+1, layer_means[i], layer_stds[i]))

# 9. Grafiksel gösterim
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), layer_means, 'o-b')
plt.title("Layer Means")
plt.xlabel("Layer")
plt.ylabel("Mean")

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), layer_stds, 'o-r')
plt.title("Layer Standard Deviations")
plt.xlabel("Layer")
plt.ylabel("Std Dev")

plt.tight_layout()
plt.show()