import argparse
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torchvision import models
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch import optim
import os


def check_and_create_directory(directory_path):
    """
    Check if the directory exists. If not, create it.

    Args:
        directory_path (str): The path to the directory to check or create.
    """
    if not os.path.exists(directory_path):
        # If the directory doesn't exist, create it
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Main function to parse arguments and execute training
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset of images.')
    
    parser.add_argument('data_dir', type=str, help='Directory containing the training data.')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture to use (vgg16, vgg13).')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier.')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to train the model.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training.')

    # Parse arguments
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print("Error! : " + args.data_dir + " directory does not exist for training. Unable to train. (Try flower)")
        return None
    
    check_and_create_directory(args.save_dir)

    # Check torch version and CUDA status if GPU is enabled.
    print(torch.__version__)
    print(torch.cuda.is_available()) # Should return True when GPU is enabled.

    


    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),  # Resize to 256x256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),              # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize based on ImageNet values
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),              # Resize to 256x256
            transforms.CenterCrop(224),          # Center crop to 224x224
            transforms.ToTensor(),               # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),              # Resize to 256x256
            transforms.CenterCrop(224),          # Center crop to 224x224
            transforms.ToTensor(),               # Convert image to tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
        ])
    }

    # TODO: Load the datasets with ImageFolder
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
        'valid': DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False)
    }

    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    print("Completed label mapping")

    # Only support 2 types of vision models now

    if args.arch == "vgg16":
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    elif args.arch == "vgg13":
        from torchvision.models import VGG13_Weights
        model = models.vgg13(weights=VGG13_Weights.IMAGENET1K_V1)
    else:
        print(f"Architecture {args.arch} not supported. Exiting.")
        return None

    


    for param in model.parameters():
        param.requires_grad = False
        
    print("Get the number of inputs for the classifier (4096 in the case of VGG16)")
    num_features = model.classifier[0].in_features

    print("Define a new feed-forward classifier")
    classifier = nn.Sequential(
        nn.Linear(num_features, 512),  # Fully connected layer (input: 4096, output: 512)
        nn.ReLU(),                     # ReLU activation
        nn.Dropout(0.2),               # Dropout for regularization
        nn.Linear(512, 102),           # Output layer (102 categories for flowers, adjust as needed)
        nn.LogSoftmax(dim=1)           # Log-Softmax for classification
    )

    print("Replace the classifier in the VGG16 model")
    model.classifier = classifier



    print("Define the loss function and optimizer")
    criterion = nn.NLLLoss()  # Negative Log Likelihood Loss
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)  # Optimizer for the classifier

    device = torch.device('cuda')
    if args.gpu and torch.cuda.is_available():
        print(torch.__version__)
        device = torch.device('cuda')
        print("Using GPU for training.")
    else:
        device = torch.device('cpu')
        print("Using CPU for training.")
    model.to(device)
    print (device.type)

    epochs = args.epochs # 2 rounds enough
    steps = 0

    for epoch in range(epochs):
        running_loss = 0
        model.train()  
        print("Set model to training mode...")

        for images, labels in dataloaders['train']:
            steps += 1

            print(steps, device.type, "Move input and label tensors to device ")
                
            images, labels = images.to(device), labels.to(device)

            #print("Zero the parameter gradients")
            optimizer.zero_grad()

            #print("Forward pass through the model")
            logps = model(images)
            loss = criterion(logps, labels)

            #print("Backward pass and optimization")
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            
                
    # Test the model on the test set
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    accuracy = 0

    criterion = nn.NLLLoss()  # Use the same loss function as used during training

    # Disable gradient calculations to speed up inference
    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)

            # Forward pass through the model
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    # Compute the average loss and accuracy across the test set
    test_loss /= len(dataloaders['test'])
    accuracy /= len(dataloaders['test'])

    print(f"Test Loss: {test_loss:.3f}.. "
        f"Test Accuracy: {accuracy:.3f}")

    # TODO: Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'args': args.arch
    }

    torch.save(checkpoint, args.save_dir + "/" + 'vgg16_flower_classifier.pth')

if __name__ == '__main__':
    main()




