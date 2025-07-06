import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

# Create dataset
print("Creating dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=10,
    random_state=42
)

# Split data into train/validation/test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"Train set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

# Define parameters to test - YOU CAN MODIFY THESE VALUES
learning_rates = [0.001, 0.01, 0.1]
regularization_alphas = [0.0001, 0.001, 0.01, 0.1]
hidden_layer_size = (50, 10)  # Fixed architecture for consistency

print(f"\nTesting parameters across 5 epochs:")
print(f"Hidden layer architecture: {hidden_layer_size}")
print(f"Learning rates: {learning_rates}")
print(f"Regularization alphas: {regularization_alphas}")
print("="*80)

# Store results for each combination
results = {}

# Test each combination
for lr in learning_rates:
    for alpha in regularization_alphas:
        param_key = f"lr_{lr}_alpha_{alpha}"
        print(f"\nTesting: Learning Rate = {lr}, Alpha = {alpha}")
        print("-" * 50)
        
        # Lists to store metrics for each epoch
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []
        
        # Train for 5 epochs, recording metrics after each epoch
        for epoch in range(1, 6):
            try:
                # Create model with max_iter = epoch to train incrementally
                model = MLPClassifier(
                    hidden_layer_sizes=hidden_layer_size,
                    learning_rate_init=lr,
                    alpha=alpha,
                    max_iter=epoch,
                    random_state=42,
                    warm_start=True if epoch > 1 else False,
                    early_stopping=False  # Disable early stopping to see all 5 epochs
                )
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Calculate predictions and probabilities
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                train_proba = model.predict_proba(X_train)
                val_proba = model.predict_proba(X_val)
                
                # Calculate accuracies
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                # Calculate losses
                train_loss = log_loss(y_train, train_proba)
                val_loss = log_loss(y_val, val_proba)
                
                # Store results
                train_accuracies.append(train_acc)
                val_accuracies.append(val_acc)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
            except Exception as e:
                print(f"Error at epoch {epoch}: {str(e)}")
                break
        
        # Store results for this parameter combination
        results[param_key] = {
            'lr': lr,
            'alpha': alpha,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'train_losses': train_losses,
            'val_losses': val_losses
        }

print("\n" + "="*80)
print("SUMMARY OF ALL COMBINATIONS:")
print("="*80)

for param_key, data in results.items():
    if len(data['val_accuracies']) == 5:  # Only show complete runs
        final_val_acc = data['val_accuracies'][-1]
        final_val_loss = data['val_losses'][-1]
        print(f"LR: {data['lr']:6.4f}, Alpha: {data['alpha']:6.4f} -> "
              f"Final Val Acc: {final_val_acc:.4f}, Final Val Loss: {final_val_loss:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
epochs = list(range(1, 6))

# Plot 1: Validation Accuracy over epochs
axes[0, 0].set_title('Validation Accuracy Across 5 Epochs')
for param_key, data in results.items():
    if len(data['val_accuracies']) == 5:
        label = f"LR: {data['lr']}, α: {data['alpha']}"
        axes[0, 0].plot(epochs, data['val_accuracies'], marker='o', label=label)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Validation Accuracy')
axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Validation Loss over epochs
axes[0, 1].set_title('Validation Loss Across 5 Epochs')
for param_key, data in results.items():
    if len(data['val_losses']) == 5:
        label = f"LR: {data['lr']}, α: {data['alpha']}"
        axes[0, 1].plot(epochs, data['val_losses'], marker='o', label=label)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Validation Loss')
axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Train vs Val Accuracy for best performing combination
best_param = max(results.keys(), 
                key=lambda k: results[k]['val_accuracies'][-1] if len(results[k]['val_accuracies']) == 5 else 0)
best_data = results[best_param]

axes[1, 0].set_title(f'Train vs Val Accuracy (Best: LR={best_data["lr"]}, α={best_data["alpha"]})')
axes[1, 0].plot(epochs, best_data['train_accuracies'], marker='o', label='Train Accuracy', color='blue')
axes[1, 0].plot(epochs, best_data['val_accuracies'], marker='s', label='Val Accuracy', color='red')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Train vs Val Loss for best performing combination
axes[1, 1].set_title(f'Train vs Val Loss (Best: LR={best_data["lr"]}, α={best_data["alpha"]})')
axes[1, 1].plot(epochs, best_data['train_losses'], marker='o', label='Train Loss', color='blue')
axes[1, 1].plot(epochs, best_data['val_losses'], marker='s', label='Val Loss', color='red')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final comparison table
print("\n" + "="*80)
print("FINAL COMPARISON TABLE:")
print("="*80)
print(f"{'Learning Rate':<12} {'Alpha':<8} {'Final Val Acc':<14} {'Final Val Loss':<14} {'Overfitting':<12}")
print("-" * 80)

for param_key, data in results.items():
    if len(data['val_accuracies']) == 5:
        final_train_acc = data['train_accuracies'][-1]
        final_val_acc = data['val_accuracies'][-1]
        final_val_loss = data['val_losses'][-1]
        overfitting = "Yes" if (final_train_acc - final_val_acc) > 0.05 else "No"
        
        print(f"{data['lr']:<12.4f} {data['alpha']:<8.4f} {final_val_acc:<14.4f} "
              f"{final_val_loss:<14.4f} {overfitting:<12}")

print("\n" + "="*80)
print("ANALYSIS:")
print("• Higher learning rates generally converge faster but may be unstable")
print("• Higher regularization (alpha) reduces overfitting but may hurt performance")
print("• Look for combinations with stable validation metrics across epochs")
print("• Check train vs validation gap to identify overfitting")