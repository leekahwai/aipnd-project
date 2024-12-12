import argparse
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision.models import VGG16_Weights
from torchvision.models import VGG13_Weights

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    """Process an image path into a format suitable for the model."""
    
    # Step 1: Open the image using PIL
    pil_image = Image.open(image)
    
    # Step 2: Resize the image so the shortest side is 256 pixels, maintaining aspect ratio
    # Thumbnail will keep the aspect ratio, but won't exceed the size (256 for the shortest side)
    if pil_image.size[0] < pil_image.size[1]:
        pil_image = pil_image.resize((256, int(256 * pil_image.size[1] / pil_image.size[0])))
    else:
        pil_image = pil_image.resize((int(256 * pil_image.size[0] / pil_image.size[1]), 256))
    
    # Step 3: Center-crop the image to 224x224 pixels
    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    pil_image = pil_image.crop((left, top, right, bottom))
    
    # Step 4: Convert the image to a Numpy array and scale the pixel values to be between 0 and 1
    np_image = np.array(pil_image) / 255.0
    
    # Step 5: Normalize the image by subtracting the mean and dividing by the standard deviation
    # Means and standard deviations for each color channel (Red, Green, Blue)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Step 6: Reorder the dimensions so that the color channel is first
    # The current order is (height, width, color), but PyTorch expects (color, height, width)
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert to a PyTorch tensor
    image_tensor = torch.from_numpy(np_image).float()
    
    return image_tensor

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, gpu, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Step 1: Process the image using the process_image function
    image = process_image(image_path)
    
    # Step 2: Move the image tensor to the device (GPU or CPU)
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)
    
    # Add a batch dimension (model expects batches of images)
    image = image.unsqueeze(0).to(device)
    
    # Set model to evaluation mode and turn off gradients
    model.eval()
    with torch.no_grad():
        # Step 3: Get the model's predictions (forward pass)
        output = model(image)
        
        # Step 4: Apply softmax to get probabilities
        ps = torch.exp(output)
        
        # Step 5: Get the top k probabilities and indices
        top_p, top_class = ps.topk(topk, dim=1)
        
        # Convert probabilities and indices to lists
        probs = top_p.cpu().numpy().squeeze().tolist()
        classes = top_class.cpu().numpy().squeeze().tolist()
    
    # Step 6: Invert the class_to_idx dictionary to map indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Step 7: Map the model's predicted indices to the actual class labels
    predicted_classes = [idx_to_class[i] for i in classes]
    
    return probs, predicted_classes




    
    

def do_predictions(image_path, model, cat_to_name, gpu, topk=5):
    """Display image and the top 5 class predictions."""
    
    # Predict the top 5 classes
    probs, classes = predict(image_path, model, gpu, topk)
    
    # Convert class indices to flower names using cat_to_name mapping
    flower_names = [cat_to_name[str(cls)] for cls in classes]
    
   
    return flower_names, probs

# Main function to parse arguments and execute training
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset of images.')
    
    parser.add_argument('image_to_predict', type=str, help='Image to predict')
    parser.add_argument('--top_k', type=str, default='.', help='Directory to save checkpoints.')
    parser.add_argument('checkpoint', type=str, default='save_dir/vgg16_flower_classifier.pth', help='Model to use for prediction.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Category of names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    # Parse arguments
    args = parser.parse_args()

    # Check torch version and CUDA status if GPU is enabled.
    print(torch.__version__)
    print(torch.cuda.is_available()) # Should return True when GPU is enabled.

    if args.gpu and torch.cuda.is_available():
        print(torch.__version__)
        device = torch.device('cuda')
        print("Using GPU for training.")
    else:
        device = torch.device('cpu')
        print("Using CPU for training.")

    # TODO: Write a function that loads a checkpoint and rebuilds the model
    # Load the saved checkpoint
    checkpoint = torch.load(args.checkpoint)

    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    if checkpoint['args'] == 'vgg13':
        model = models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
        print("vgg13 model is being used")
    elif checkpoint['args'] == 'vgg16':
        print("vgg16 model is being used")
    else:
        print ("" + checkpoint['args'] + " is not supported")
        return None

    # Freeze the feature layers (so we don't retrain them)
    for param in model.parameters():
        param.requires_grad = False

    # Rebuild the classifier from the saved checkpoint
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']

    # Load the model's state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Optionally, load the optimizer's state if you plan to continue training
    optimizer = torch.optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    cat_to_name = None
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    flower, probs = do_predictions(args.image_to_predict, model, cat_to_name, args.top_k)
    
    print(flower, probs)
    

if __name__ == '__main__':
    main()