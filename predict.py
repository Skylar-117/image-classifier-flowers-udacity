import os
import numpy as np
import pandas as pd

import torch
import json
import PIL
from torchvision import models, transforms

from train import device

import argparse
import sys

def arg_parser():
    '''
    Takes in command-line arguments and parses them for usage of our Python functions.
    '''
    
    parser = argparse.ArgumentParser(description='Image Classifier Prediction Params')
    
    parser.add_argument('--gpu', 
                        type=str, 
                        default='Y',
                        help='Use GPU (Y for Yes; N for No). Default is Y.')

    parser.add_argument('--checkpoint',
                        type=str,
                        default='checkpoint.pth',
                        help='Path for model checkpoint created using train.py. Default is \'./checkpoint.pth\'.')

    parser.add_argument('--image',
                        type=str,
                        help='Path for image to be predicted.')

    parser.add_argument('--topk', 
                        type=int, 
                        default=5,
                        help='Top K predictions to show. Default is 5.')

    args = parser.parse_args()

    return(args)

def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    
    Parameters
    ==========
    image_path: str, file paths of the images 
    '''
    image1=Image.open(image_path)
    
    # TODO: Process a PIL image for use in a PyTorch model
    proc = transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])])
    image1 = proc(image1)
    return image1

def load_model(checkpoint_path, device):
    """
    Loads a checkpoint and rebuild the model
    
    Parameters
    ==========
    checkpoint_path: str, name of the checkpoint file returned from save() function
    device: device object returned from device() function
    """
    checkpoint = torch.load(checkpoint_path)
    architecture = checkpoint['architecture']
    
    if(architecture=='vgg16'):
        model = models.vgg16(pretrained=True)
        model = model.to(device)
    else:
        print('Unsupported architecture. Only VGG16 is available.')
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def predict(model, image_path, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    For here, CPU mode is used for prediction.
    '''
    model.eval()
    model.to('cpu')
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(image)
        
    probabilities = torch.exp(output)
    
    if(topk):
        topk = topk
    else:
        print('No Top K specified, will use the default value - 5')
        topk = 5
    
    topk_probabilities, topk_labels = probabilities.topk(topk)
    
    labels_list = test_labels.squeeze().tolist()
    probabilities_list = test_probabilities.squeeze().tolist()
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    text_labels_list = [cat_to_name[str(label)] for label in labels_list]
    
    pred_lists = dict(zip(labels_list, probabilities_list))
    
    return pred_lists

def main():
    
    args = arg_parser()
    
    if(args.gpu):
        gpu = args.gpu
    else:
        print('GPU mode not specified, will use the default value - "Y"')
        gpu = "Y"
    device = device(gpu)
    
    # Model loading:
    if(args.checkpoint):
        model = load_model(args.checkpoint, device)
    else:
        model = load_model('checkpoint.pth', device)

    # Image processing:
    image = process_image(args.image)

    # Predictions:
    predictions = predict(model, image, device, args.topk)
    print(predictions)

if __name__ == '__main__':
    main()