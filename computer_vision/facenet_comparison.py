import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

# Initialize the MTCNN module for face detection and the InceptionResnetV1 module for face embedding.
mtcnn = MTCNN(image_size=160, keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding_and_face(image_path):
    """Load an image (from URL or local path), detect the face, and return the embedding and face."""
    try:
        if image_path.startswith("http"):  # Check if it's a URL
            response = requests.get(image_path)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type')
            if 'image' not in content_type:
                raise ValueError(f"URL does not point to an image: {content_type}")
            image = Image.open(BytesIO(response.content)).convert("RGB")
        elif os.path.isfile(image_path):  # Check if it's a local file
            image = Image.open(image_path).convert("RGB")
        else:
            raise ValueError(f"Invalid image path: {image_path}")
    except Exception as e:
        print(f"Error loading image from {image_path}: {e}")
        return None, None

    faces, probs = mtcnn(image, return_prob=True)
    if faces is None or len(faces) == 0:
        return None, None

    embedding = resnet(faces[0].unsqueeze(0))
    return embedding, faces[0]

def tensor_to_image(tensor):
    """Convert a normalized tensor to a valid image array."""
    image = tensor.permute(1, 2, 0).detach().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype('uint8')
    return image

def find_most_similar(target_image_path, candidate_image_paths):
    """Find the most similar image to the target image from a list of candidate images."""
    target_emb, target_face = get_embedding_and_face(target_image_path)
    if target_emb is None:
        raise ValueError("No face detected in the target image.")

    highest_similarity = float('-inf')
    most_similar_face = None
    most_similar_image_path = None

    candidate_faces = []
    similarities = []

    for candidate_image_path in candidate_image_paths:
        candidate_emb, candidate_face = get_embedding_and_face(candidate_image_path)
        if candidate_emb is None:
            similarities.append(None)
            candidate_faces.append(None)
            continue

        similarity = torch.nn.functional.cosine_similarity(target_emb, candidate_emb).item()
        similarities.append(similarity)
        candidate_faces.append(candidate_face)

        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_face = candidate_face
            most_similar_image_path = candidate_image_path

    # Visualization
    plt.figure(figsize=(12, 8))

    # Display target image
    plt.subplot(2, len(candidate_image_paths) + 1, 1)
    plt.imshow(tensor_to_image(target_face))
    plt.title("Target Image")
    plt.axis("off")

    # Display most similar image
    if most_similar_face is not None:
        plt.subplot(2, len(candidate_image_paths) + 1, 2)
        plt.imshow(tensor_to_image(most_similar_face))
        plt.title("Most Similar")
        plt.axis("off")

    # Display all candidate images with similarity scores
    for idx, (candidate_face, similarity) in enumerate(zip(candidate_faces, similarities)):
        plt.subplot(2, len(candidate_image_paths) + 1, idx + len(candidate_image_paths) + 2)
        if candidate_face is not None:
            plt.imshow(tensor_to_image(candidate_face))
            plt.title(f"Score: {similarity * 100:.2f}%")
        else:
            plt.title("No Face")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    if most_similar_image_path is None:
        raise ValueError("No faces detected in the candidate images.")

    return most_similar_image_path, highest_similarity

# Add a JPEG image of yourself to this directory and replace the image_url_target below with the path to your local image: 
image_url_target = 'https://d1mnxluw9mpf9w.cloudfront.net/media/7588/4x3/1200.jpg'
# Play around with the comparison images! Add additional celebrity URLs and/or local images of whoever you'd like:
candidate_image_urls = [
    'https://beyondthesinglestory.wordpress.com/wp-content/uploads/2021/04/elon_musk_royal_society_crop1.jpg',
    'https://cdn.britannica.com/56/199056-050-CCC44482/Jeff-Bezos-2017.jpg',
    'https://cdn.britannica.com/45/188745-050-7B822E21/Richard-Branson-2003.jpg'
]

most_similar_image, similarity_score = find_most_similar(image_url_target, candidate_image_urls)
print(f"The most similar image is: {most_similar_image}")
print(f"Similarity score: {similarity_score * 100:.2f}%")