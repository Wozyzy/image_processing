import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import numpy as np
import tensorflow as tf

# Model sınıfını tanımlayalım (mnist_example.py'dan aynı model)
# Model sinifini tanimlayalim (mnist_example.py'dan ayni model)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# Modeli yükle
# Modeli yukle
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("your_digit.png", weights_only=True))
model.eval()

def preprocess_image(image_path):
    # Görüntüyü yükle
    # Goruntuyu yukle
    image = Image.open(image_path).convert('L')  # Gri tonlamaya çevir
    image = Image.open(image_path).convert('L')  # Gri tonlamaya cevir
    
    # Görüntüyü 28x28 boyutuna getir
    # Goruntuyu 28x28 boyutuna getir
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST veri setinin ortalama ve standart sapması
    ])
    
    # Görüntüyü dönüştür
    # Goruntuyu donustur
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Batch boyutu ekle

def predict_digit(image_path):
    # Görüntüyü ön işle
    # Goruntuyu on isle
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Tahmin yap
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(1).item()
    
    return prediction

# Test etmek için
# Test etmek icin
if __name__ == "__main__":
    # Görüntü yolunu buraya yazın
    # Goruntu yolunu buraya yazin
    image_path = "your_digit.png"  # Görüntü adını güncelledik
    image_path = "your_digit.png"  # Goruntu adini guncelledik
    try:
        prediction = predict_digit(image_path)
        print(f"Tahmin edilen rakam: {prediction}")
    except Exception as e:
        print(f"Hata olustu: {e}")
        print("Lutfen gecerli bir goruntu dosyasi yolu belirttiginizden emin olun.") 