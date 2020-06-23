import os
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from torchvision import datasets, models, transforms

import argparse
from collections import OrderedDict

def arg_parser():
    '''
    Get the command line input into the scripts via argparser module
    '''
    
    parser = argparse.ArgumentParser(description='Image Classifier Parameters')
    
    parser.add_argument('--architecture', 
                        type=str, 
                        help='Architecture and model from torchvision.models as strings: vgg16 and densenet121 supported.')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Learning rate of the Neural Network. Default is 0.001.')
    parser.add_argument('--hidden_size', 
                        type=int, 
                        help='Size of the hidden layer of the Neural Network. Default is 1500.')
    parser.add_argument('--dropout',
                       type=float,
                       help='Dropout value for the dropout layer. Default is 0.5.')
    parser.add_argument('--output_size',
                       type=int,
                       help='Size of the network output. Default is 102.')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training Neural Network. Default is 5.')
    parser.add_argument('--gpu', 
                        type=str, 
                        help='Use GPU or not, (if Y then use GPU; if N then do not use GPU). Default is Y.')

    args = parser.parse_args()
    return(args)

def load_dataset(data_dir='flowers'):
    """
    Function for loading dataset and spliting into train/valid/test three parts, also transformation is applied on training set.
    All three datasets are returned by this function.
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Apply transformations on training set, leave alone validation and testing sets:
    data_transforms = {
        "training" : transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]),
        # For validation and tesing sets, since they are the "unseen" data that used to measure the model performance, so they should not be applied by any transformations, however, resizing is stil needed.
        "validation" : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
        "testing" : transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    }
    
    # Load datasets with ImageFolder:
    image_datasets = {
        "training" : datasets.ImageFolder(train_dir, transform = data_transforms["training"]),
        "validation" : datasets.ImageFolder(valid_dir, transform = data_transforms["validation"]),
        "testing" : datasets.ImageFolder(test_dir, transform = data_transforms["testing"])
    }
    
    # Using the image datasets and the trainforms, define the dataloaders: 
    dataloaders = {
        "training" : torch.utils.data.DataLoader(image_datasets["training"], batch_size = 64, shuffle = True),
        "validation" : torch.utils.data.DataLoader(image_datasets["validation"], batch_size = 64),
        "testing" : torch.utils.data.DataLoader(image_datasets["testing"], batch_size = 64)
    }
    
    return (dataloaders['training'],
            dataloaders['validation'],
            dataloaders['testing'],
            image_datasets['training'],
            image_datasets['validation'],
            image_datasets['testing'])

def device(gpu):
    '''
    Device setting of GPU usage or not. Device variable is returned.
    
    Parameters
    ==========
    gpu: string, Y or N based on whether use the GPU.
    '''
    if(gpu=='Y'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if(device=='cpu'):
            print('CUDA not available, will use CPU...')
        else:
            print('Use GPU...')
    else: # CPU mode
        device = 'cpu'
        print('Use CPU...')

    return(device)

def network(device, architecture='vgg16', learning_rate=0.001, hidden_size=1500, dropout=0.5, output_size=102):
    '''
    Network structure.
    
    Parameters
    ==========
    device: device object returned from function device()
    architecture: str, here only vgg16 is available
    learning_rate: float, learning rate from the model parameter, default 0.001
    hidden_size: int, only 1 hidden layer here, and this is the size of the input of the hidden layer, default 1500
    dropout: float, dropout value for dropout layer, default 0.5
    output_size: int, default is 102 and this is not changable since 102 is the number of classes
    '''
    model = models.vgg16(pretrained=True)
    model.name = architecture
    input_size = model.classifier[0].in_features
        
    if(hidden_size):
        hidden_size = hidden_size
    else:
        print('Number of hidden layer input size not specified, will use the default value - 1500')
        hidden = 1500
        
    if(learning_rate):
        learning_rate = learning_rate
    else:
        print('Learning rate not specified, will use the default value - 0.001')
        learning_rate = 0.001
        
    if(dropout):
        dropout = dropout
    else:
        print('Dropout value not specified, will use the default value - 0.5')
        dropout = 0.5

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_size)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(dropout)),
        ('fc2', nn.Linear(hidden_size, output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    model.to(device)

    # Criterion and optimizer settings:
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return(model, criterion, optimizer)

def train(model, epochs, learning_rate, criterion, optimizer, training_loader, validation_loader, device):

    """Train the NN.
    
    Parameters
    ==========
    model: network object returned from network() function, this is the network architecture
    epochs: int, number of epochs
    learning_rate: float, learning rate value
    criterion: loss function to use, default is NLLLoss() returned from network() function
    optimizer: optimize method, Adam is used here, it is returned from network() function
    training_loader: dataloaders for training
    validation_loader: dataloaders for validation
    """
    steps = 0
    model.to(device)
    
    if(epochs):
        epochs = epochs
    else:
        print('No epochs specified, will use dafault value - 5')
        epochs = 5

    # Start the training process:
    print("Starting training ...")
    for e in range(epochs):
        running_loss = 0
        for inputs_train, labels_train in training_loader:
            step += 1
            inputs_train = inputs_train.to(device)
            labels_train = labels_train.to(device)

            # Zero the gradient:
            optimizer.zero_grad()
            output = model.forward(inputs_train) # Do the forward calculation
            loss = criterion(output, labels_train) # Calculate cross entropy loss
            loss.backward() # Calculate the gradient with respect to each weight backward
            optimizer.step() # Update the weights(bias is included)

            running_loss += loss.item() # This tracks the training loss over all the epochs

            # Every 'print_every' steps, validate the results:
            if(step%print_every==0):

                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs_valid, labels_valid in validation_loader:
                        inputs_valid = inputs_valid.to(device)
                        labels_valid = labels_valid.to(device)

                        output = model.forward(inputs_valid)
                        loss_valid = criterion(output, labels_valid)

                        valid_loss += loss_valid.item() # This tracks the valid loss over all the epochs

                        # Calculate accuracy
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_valid.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch: {e+1}/{epochs}, "
                      f"Train loss: {running_loss/print_every:.3f}, "
                      f"Validation loss: {valid_loss/len(validation_loader):.3f}, "
                      f"Validation accuracy: {accuracy/len(validation_loader):.3f}")

                running_loss = 0
                model.train()

    print("Finished training!")
    return model

def test(model, criterion, testing_loader):
    """Function for validating the result on test set.
    
    Parameters
    ==========
    model: network object returned from network() function
    criterion: loss function returned from network() function
    testing_loader: dataloader for testing
    """
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testing_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model.forward(inputs)
            loss = criterion(output, labels)

            test_loss += loss.item() # This tracks the valid loss over all the epochs

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Accuracy on test set is: {accuracy/len(testing_loader):.3f}")

def save(model, train_data=image_datasets["training"], epochs, architecture):
    '''Saves the model checkpoint to the given path.
    '''
    model.class_to_idx = train_data.class_to_idx
    
    if(epochs):
        epochs = epochs
    else:
        print("Epochs not specified, will use the default value - 5")
        epochs = 5

    checkpoint = {"architecture": "vgg16",
                  "input_size": model.classifier.fc1.in_features,
                  "hidden_input_size": model.classifier.fc1.out_features,
                  "output_size": model.classifier.fc2.out_features,
                  "state_dict": model.state_dict(), # Holds all the weights and biases
                  "learning_rate": learning_rate,
                  "epochs": epochs,
                  "optimizer": optimizer.state_dict(),
                  "class_to_idx": model.class_to_idx
                 }

    torch.save(checkpoint, "checkpoint.pth")
    print(f"Model saved to {"checkpoint.pth"}")
    
def main():
    """Here is the function to run the entire training process, it gathers all steps/function orderly to go through the trainig process.
    """
    args = arg_parser()
    if(args.gpu):
        gpu = args.gpu
    else:
        print("GPU mode not specified, will use the default value - Use GPU")
        gpu = "Y"
    # Device setting:
    device = device(gpu)
    
    # Prepare the datasets and dataloaders:
    train_loader, valid_loader, test_loader, train_data, valid_data, test_data = load()
    
    # Model architects, criterion and optimizer:
    model, criterion, optimizer = network(device=device,
                                          architecture=args.architecture,
                                          learning_rate=args.learning_rate,
                                          hidden_size=args.hidden_size,
                                          dropout=args.dropout,
                                          output_size=args.output_size)
    
    # Train the model:
    model = train(model=model,
                  train_loader=train_loader,
                  valid_loader=valid_loader,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  epochs=args.epochs)
    
    # Validate the model performance on the test set:
    test(model=model, test_loader=test_loader, device=device)
    
    # Save model checkpoint:
    save(model=model, train_data=train_data, epochs=args.epochs, architecture=args.architecture)

if __name__ == '__main__':
    main()