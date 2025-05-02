# simple_gan.py

A minimal PyTorch implementation of a Generative Adversarial Network (GAN) for educational use. This script trains a GAN on the MNIST dataset and generates handwritten digit images.

## Overview

-   **Generator**: Converts random noise (latent vector) into a fake image.
-   **Discriminator**: Evaluates an image (real or fake) and predicts the probability it’s real.
-   **Adversarial Training**:
    1. Train Discriminator to tell real and fake images apart.
    2. Train Generator to fool the Discriminator into labeling fake images as real.

## Requirements

-   Python 3.x
-   PyTorch
-   torchvision

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install torch torchvision
```

## Usage

```bash
python simple_gan.py
```

## Output

-   A folder named `images/` containing generated image grids saved at set intervals.

---

Feel free to tweak hyperparameters in `simple_gan.py` and rerun to see how they affect generated images.
