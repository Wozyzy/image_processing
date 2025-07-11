import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("Veri donusumu ayarlaniyor...")
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Veri yükleniyor...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Sadece 2000 ornek alalim (20 batch)
subset_indices = list(range(1000))
train_subset = torch.utils.data.Subset(trainset, subset_indices)
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=100, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

print("Model kuruluyor...")
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanilan cihaz: {device}")

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

train_losses = []
accuracies = []

print("Egitime basliyoruz...\n")

for epoch in range(3):  
    print(f"Epoch {epoch+1} basladi...")
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}/{len(trainloader)} | Anlik Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(trainloader)
    train_losses.append(avg_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    accuracies.append(acc)
    print(f"Epoch {epoch+1} tamamlandi - Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%\n")

