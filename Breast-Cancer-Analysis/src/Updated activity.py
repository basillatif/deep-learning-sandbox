import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN with improvements
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Batch normalization Layer
        self.bn1 = nn.BatchNorm2d(32)
        # Fully connected Layer
        self.fc1 = nn.Linear(32 * 28 * 28, 10)  # Example for MNIST (28x28 images)

    def forward(self, x):
        # Apply convolution, batch normalization, and ReLU activation
        x = torch.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# Instantiate the model
model = ImprovedCNN()

# Apply He initialization to weights
for layer in model.modules():
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Use Adam optimizer for faster convergence
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Loss function
criterion = nn.CrossEntropyLoss()

# Print the model architecture
print(model)
