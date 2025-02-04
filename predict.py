import argparse
import torch
from torchvision import models
from PIL import Image
import json
import numpy as np

from train import load_checkpoint  # Ensure train.py contains a function to load the model

def process_image(image_path):
    """Scales, crops, and normalizes a PIL image for a PyTorch model"""
    image = Image.open(image_path)
    
    # Resize the shortest side to 256 keeping the aspect ratio
    aspect_ratio = image.size[0] / image.size[1]
    if image.size[0] < image.size[1]:
        image = image.resize((256, int(256 / aspect_ratio)))
    else:
        image = image.resize((int(256 * aspect_ratio), 256))
    
    # Center crop to 224x224
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    
    # Convert to NumPy array and normalize
    np_image = np.array(image) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose to match PyTorch tensor shape
    np_image = np_image.transpose((2, 0, 1))
    return torch.tensor(np_image, dtype=torch.float32)

def predict(image_path, model, topk=5, category_names=None, device='cpu'):
    """Predict the class (or classes) of an image using a trained deep learning model."""
    model.to(device)
    model.eval()
    
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        probabilities, indices = torch.exp(output).topk(topk)
    
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx.item()] for idx in indices[0]]
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]
    
    return probabilities[0].tolist(), classes

def main():
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to saved model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    
    model = load_checkpoint(args.checkpoint)
    probabilities, classes = predict(args.image_path, model, args.top_k, args.category_names, device)
    
    print(f"Top {args.top_k} predictions:")
    for prob, cls in zip(probabilities, classes):
        print(f"{cls}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
