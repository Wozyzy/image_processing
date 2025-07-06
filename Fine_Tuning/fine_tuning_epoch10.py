import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
subset_indices = list(range(100))  # Simulate small dataset (first 100 samples)
train_subset = torch.utils.data.Subset(trainset, subset_indices)
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=10, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet18 model
model = models.resnet18(weights=True)
model = model.to(device)

# Freeze all layers (for fine-tuning)
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for CIFAR-10 (10 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)
model.fc.requires_grad = True
model.fc = model.fc.to(device)

# Only the final layer will be trainable
for param in model.fc.parameters():
    param.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()
        total_samples += images.size(0)
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")

print("Fine-tuning completed for 10 epochs.")