import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN with intentional flaws

class FlawedCNN(nn.Module):

    def __init__(self):
        super(FlawedCNN, self).__init__()
        # Convolutional Layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Fully connected Layer
        self.fc1 = nn.Linear(32 * 28 * 28, 10)  # Example for MNIST (28x28 images)

    def forward(self, x):
        # Flaw: No activation function after conv1
        x = self.conv1(x)
        # Flaw: No batch normalization
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x

# Instantiate the model
model = FlawedCNN()

# Flaw: Poor initialization (default PyTorch initialization is used)
# Flaw: Suboptimal optimizer (SGD with no momentum)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loss function
criterion = nn.CrossEntropyLoss()

# Print the model architecture
print(model)


print("No Activation Function after Convolution: This can affect learning because the convolution output isn’t passed through a non-linearity like ReLU.") 
      
print("No Batch Normalization: This may lead to slower convergence and instability in training.SGD with No Momentum: This makes optimization slower and less efficient.")