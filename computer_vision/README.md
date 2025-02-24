# FaceNet-PyTorch Image Similarity Comparison

## Overview
This project uses **FaceNet-PyTorch** to compare face embeddings and find the most visually similar face among a set of candidate images. The script loads images, detects faces, extracts embeddings, and calculates similarity scores.

## Installation and Setup

### 1. Install Python 3.11
To ensure compatibility with PyTorch and FaceNet, install an appropriate Python version:
```sh
brew install python@3.11
```

### 2. Create and Set Up a Virtual Environment
Navigate to your workspace and set up a virtual Python environment:
```sh
mkdir deeplearning_facenet
cd deeplearning_facenet
python3.11 -m venv .venv
source .venv/bin/activate
```
You’ll know the virtual environment is active when your terminal prompt starts with `(.venv)`.

### 3. Install Required Dependencies
Upgrade `pip` and install necessary libraries:
```sh
pip install --upgrade pip
pip install --no-cache-dir torch facenet-pytorch requests matplotlib
```

## Running the Script

### 1. Create the Python Script
Copy and paste the provided Python code into a new file named `facenet_comparison.py` inside the `deeplearning_facenet` directory.

### 2. Run the Script
Execute the script from the terminal (ensure the virtual environment is active):
```sh
python test.py
```

The script will display visual outputs and print similarity scores in the command-line output.

## Expected Output
- The terminal will display the most similar image's URL and its similarity score.
- The script will generate a visualization comparing the target image with all candidate images, highlighting the most similar one.

## Notes
- Ensure you run all commands within the `.venv` environment.
- If you encounter any dependency issues, verify that you are using Python 3.11 within the virtual environment.
- The script currently fetches images from URLs; to use local images, modify the `get_embedding_and_face` function to load images from disk using `PIL.Image.open("path/to/image.jpg")`.

## Troubleshooting
- **Issue:** `ModuleNotFoundError: No module named 'facenet_pytorch'`  
  **Solution:** Ensure the virtual environment is active before running the script:  
  ```sh
  source .venv/bin/activate
  ```
- **Issue:** `torch.cuda.is_available() == False` (slow performance)  
  **Solution:** If using a GPU, ensure PyTorch is installed with CUDA support. Otherwise, the script will run on CPU, which may be slower.

---
Now you're ready to run face similarity comparisons using FaceNet! 🚀

## Credits
This project is inspired by and builds upon concepts from [this FaceNet tutorial](https://dev.to/edgaras/face-recognition-with-facenet-ha8) by Edgaras on Dev.to.




