import argparse
import torch
from torchvision import models
from model_utils import load_data, build_classifier, save_checkpoint

def train_model(data_dir, arch='vgg16', save_dir='.', learning_rate=0.01, hidden_units=512, epochs=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load data
    dataloaders, image_datasets = load_data(data_dir)
    
    # Load pre-trained model
    model = models.__dict__[arch](pretrained=True)
    model = build_classifier(model, hidden_units)
    model.to(device)
    
    # Define loss and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {running_loss/len(dataloaders['train']):.3f}")
    
    # Save model checkpoint
    save_checkpoint(model, optimizer, epochs, save_dir, image_datasets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoint')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in classifier')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.arch, args.save_dir, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
