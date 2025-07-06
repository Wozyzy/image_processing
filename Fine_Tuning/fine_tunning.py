import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim

# CIFAR-10 veri kümesi yükle
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_indices = list(range(100))  # Küçük veri simülasyonu (ilk 100 örnek)
train_subset = torch.utils.data.Subset(trainset, subset_indices)
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=10, shuffle=True)

# Önceden eğitilmiş ResNet18 modeli yükle
model = models.resnet18(pretrained=True)

# Tüm katmanları dondur (fine-tuning yapılacak)
for param in model.parameters():
    param.requires_grad = False

# Son katmanı CIFAR-10 için değiştir (10 sınıf)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# Sadece son katmanı eğitilebilir hale getir
for param in model.fc.parameters():
    param.requires_grad = True

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Eğitim döngüsü (1 epoch örnek)
model.train()
for images, labels in trainloader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

print("Fine-tuning işlemi örnek olarak 1 epoch tamamlandı.")