import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_image(image_path):
    """Load an image and convert it to grayscale numpy array."""
    img = Image.open(image_path).convert('L')
    return np.array(img)

def pad_image(image, kernel_size):
    """Pad the image with zeros to handle edges."""
    pad_height = kernel_size[0] // 2
    pad_width = kernel_size[1] // 2
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

def manual_convolution(image, kernel):
    """
    Perform manual convolution of an image with a given kernel.
    
    Parameters:
    image (numpy.ndarray): Input image as a 2D numpy array
    kernel (numpy.ndarray): Convolution kernel as a 2D numpy array
    
    Returns:
    numpy.ndarray: Convolved image
    """
    # Get image and kernel dimensions
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Pad the image to handle edges
    padded_image = pad_image(image, kernel.shape)
    
    # Initialize output image
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            # Extract the region of interest
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Perform element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    return output

def visualize_convolution(original, convolved, kernel, title):
    """Visualize original image, kernel, and convolution result."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Display kernel
    ax2.imshow(kernel, cmap='gray')
    ax2.set_title('Kernel')
    ax2.axis('off')
    
    # Display convolved image
    ax3.imshow(convolved, cmap='gray')
    ax3.set_title('Convolved Image')
    ax3.axis('off')
    
    plt.suptitle(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load image
    image_path = "your_image.png"  # Replace with your image path
    image = load_image(image_path)
    
    # Define some example kernels
    # Edge detection kernel
    edge_kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ])
    
    # Gaussian blur kernel
    gaussian_kernel = np.array([
        [1/16, 1/8, 1/16],
        [1/8,  1/4, 1/8],
        [1/16, 1/8, 1/16]
    ])
    
    # Sharpen kernel
    sharpen_kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])
    
    # Perform convolutions
    edge_result = manual_convolution(image, edge_kernel)
    gaussian_result = manual_convolution(image, gaussian_kernel)
    sharpen_result = manual_convolution(image, sharpen_kernel)
    
    # Visualize results
    visualize_convolution(image, edge_result, edge_kernel, "Edge Detection")
    visualize_convolution(image, gaussian_result, gaussian_kernel, "Gaussian Blur")
    visualize_convolution(image, sharpen_result, sharpen_kernel, "Image Sharpening")
