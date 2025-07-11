import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset # Subset için
import matplotlib.pyplot as plt
import numpy as np # Rastgele alt küme seçimi için

# 1. Veri Kümesi Hazırlığı (MNIST)
# Görüntüleri Tensor'a çeviriyoruz
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST için ortalama ve std
])

# MNIST veri kümesini indir
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Veri kümesinin daha küçük bir alt kümesini seçelim (M2 için eğitimi hızlandırmak amacıyla)
# Örneğin, eğitim için sadece 5000 örnek, test için 1000 örnek
num_train_samples = 5000
num_test_samples = 1000

# Rastgele indeksler seçerek alt küme oluştur
train_indices = np.random.choice(len(train_dataset), num_train_samples, replace=False)
test_indices = np.random.choice(len(test_dataset), num_test_samples, replace=False)

train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

# Veri yükleyiciler (DataLoader)
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=1000, shuffle=False)

# Cihaz belirleme (GPU varsa CUDA, yoksa CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim için kullanılacak cihaz: {device}")

# 2. Model Tanımları

# Basit Evrişimli Blok
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels) # Batch Normalizasyon ekleyelim ki daha stabil olsun
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Şemadaki gibi 2 Conv + ReLU (ilki için) + Batch Norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Kısayol bağlantısı: Eğer boyutlar uyumsuzsa, 1x1 evrişim ile boyut ayarı
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x # x'i kısayol için sakla
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity) # F(x) + x
        out = self.relu(out) # Son ReLU
        return out


# Düz Sinir Ağı (PlainNet)
class PlainNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PlainNet, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),  # Giriş 1 kanal (gri tonlama), Çıkış 32 kanal
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256), # Biraz derinleştirelim ki degradasyonu görelim
            nn.AdaptiveAvgPool2d((1, 1)) # Global Ortalama Havuzlama
        )
        self.classifier = nn.Linear(256, num_classes) # Tam bağlantılı katman

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Residual Ağ (ResidualNet)
class ResidualNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResidualNet, self).__init__()
        # İlk katman genellikle düzdür
        self.conv1 = ConvBlock(1, 32)
        
        # Residual Bloklar
        self.layer1 = ResidualBlock(32, 64)
        self.layer2 = ResidualBlock(64, 128)
        self.layer3 = ResidualBlock(128, 256, stride=2) # Boyut küçültme
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 3. Eğitim Fonksiyonu
def train_model(model, train_loader, test_loader, epochs=5, model_name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.to(device) # Modeli cihaza taşı
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"\n--- {model_name} Eğitimi Başlıyor ---")
    for epoch in range(epochs):
        model.train() # Modeli eğitim moduna al
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device) # Verileri cihaza taşı

            optimizer.zero_grad() # Gradyanları sıfırla
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # Geri yayılım
            optimizer.step() # Ağırlıkları güncelle

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        # Test Aşaması
        model.eval() # Modeli değerlendirme moduna al
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad(): # Gradyan hesaplamayı kapat
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    
    print(f"--- {model_name} Eğitimi Tamamlandı ---")
    return train_losses, test_losses, train_accuracies, test_accuracies

# 4. Modelleri Eğit ve Karşılaştır

# PlainNet'i eğit
plain_model = PlainNet(num_classes=10)
plain_train_losses, plain_test_losses, plain_train_accuracies, plain_test_accuracies = \
    train_model(plain_model, train_loader, test_loader, epochs=10, model_name="PlainNet") # Epoch sayısını biraz artırabiliriz

# ResidualNet'i eğit
residual_model = ResidualNet(num_classes=10)
residual_train_losses, residual_test_losses, residual_train_accuracies, residual_test_accuracies = \
    train_model(residual_model, train_loader, test_loader, epochs=10, model_name="ResidualNet")


# 5. Sonuçları Görselleştirme
epochs_range = range(1, len(plain_train_losses) + 1)

plt.figure(figsize=(12, 5))

# Kayıp Grafiği
plt.subplot(1, 2, 1)
plt.plot(epochs_range, plain_train_losses, label='PlainNet Train Loss')
plt.plot(epochs_range, plain_test_losses, label='PlainNet Test Loss', linestyle='--')
plt.plot(epochs_range, residual_train_losses, label='ResidualNet Train Loss')
plt.plot(epochs_range, residual_test_losses, label='ResidualNet Test Loss', linestyle='--')
plt.title('Eğitim ve Test Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

# Doğruluk Grafiği
plt.subplot(1, 2, 2)
plt.plot(epochs_range, plain_train_accuracies, label='PlainNet Train Accuracy')
plt.plot(epochs_range, plain_test_accuracies, label='PlainNet Test Accuracy', linestyle='--')
plt.plot(epochs_range, residual_train_accuracies, label='ResidualNet Train Accuracy')
plt.plot(epochs_range, residual_test_accuracies, label='ResidualNet Test Accuracy', linestyle='--')
plt.title('Eğitim ve Test Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()