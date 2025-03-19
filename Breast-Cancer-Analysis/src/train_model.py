import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import random

# Business Context
print(" Business Use Case: Early Detection of Breast Cancer")
print("""
In the healthcare industry, early detection of diseases like breast cancer can save lives and reduce treatment costs.
This demo uses a neural network to classify breast tumors as malignant or benign based on features like tumor size, shape, and texture.
The goal is to assist doctors in making faster and more accurate diagnoses, improving patient outcomes.
""")

# Load dataset
df = pd.read_csv("data/breast_cancer.csv")
X = df.iloc[:, :-1].values  # Features
y = df["label"].values  # Target (0: Malignant, 1: Benign)


# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Define the Neural Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Compare Optimizers
optimizers = ["SGD", "Adam", "RMSprop"]
results = {}

for name in optimizers:
    print(f"\n🔹 Training with {name} optimizer")
    model = NeuralNet(input_size=X.shape[1], hidden_size=16, output_size=2)

    # Assign optimizer dynamically
    if name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()
    loss_values = []

    # Training loop
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    results[name] = loss_values

# Plot Loss Curves
plt.figure(figsize=(8, 5))
for name, loss in results.items():
    plt.plot(range(1, 51), loss, label=name)

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison for Different Optimizers")
plt.legend()
plt.show()

# Test Model Accuracy
with torch.no_grad():
    test_outputs = model(X_test)
    _, predictions = torch.max(test_outputs, 1)
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)

print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print(" Classification Report:")
print(classification_report(y_test, predictions, target_names=['Malignant', 'Benign']))

# Select 5 Random Test Samples
indices = random.sample(range(len(X_test)), 5)
sample_features = X_test[indices]
sample_labels = y_test[indices]

# Predict Using the Trained Model
with torch.no_grad():
    sample_outputs = model(sample_features)
    _, sample_predictions = torch.max(sample_outputs, 1)

# Print Results
print("\n Sample Predictions:")
for i in range(len(indices)):
    print(f"Sample {i+1}: Actual: {'Malignant' if sample_labels[i] == 0 else 'Benign'}, Predicted: {'Malignant' if sample_predictions[i] == 0 else 'Benign'}")

