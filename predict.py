import os
import numpy as np
import pandas as pd

import torch
import json
from PIL import Image
from torchvision import models, transforms

from train import device_setting

import argparse

def arg_parser():
    '''
    Takes in command-line arguments and parses them for usage of our Python functions.
    '''
    
    parser = argparse.ArgumentParser(description='Image Classifier Prediction Params')
    
    parser.add_argument('--gpu', default='Y', type=str, help='Use GPU (Y for Yes; N for No). Default is Y.')

    parser.add_argument('--checkpoint', default='checkpoint.pth', type=str, help='Path for model checkpoint created using train.py. Default is \'./checkpoint.pth\'.')

    parser.add_argument('--image', default='', type=str, help='Path for image to be predicted.')

    parser.add_argument('--category_names', default='cat_to_name.json', type=str, help='default category file')
    
    parser.add_argument('--topk', default=5, type=int, help='Top K predictions to show. Default is 5.')

    args = parser.parse_args()

    return(args)

def process_image(image_path):
    '''
    Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array
    
    Parameters
    ==========
    image: str, file paths of the images 
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    with Image.open(image_path) as image:
        image = transform(image).numpy()

    return image

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
    
    if(architecture == 'vgg16'):
        model = models.vgg16(pretrained=True)
        model = model.to(device)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        model = model.to(device)
    else:
        print('Architecture unavailable. Only vgg16 and densenet121 are allowed. Please use train.py to create the model.')
        
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def predict(model, image_path, category_names, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    
    For here, CPU mode is used for prediction.
    '''
    model.eval()
    
    image = torch.from_numpy(image_path).float()
    image = torch.unsqueeze(image, dim=0)
    
    class_to_idx = model.class_to_idx
    idx_to_class = {class_to_idx[k]: k for k in class_to_idx}
    
    if(topk):
        topk = topk
    else:
        print('No Top K specified, will use the default value - 5')
        topk = 5

    with torch.no_grad():
        output = model.forward(image.to(device))
        prediction = torch.exp(output).topk(topk)
    
    probabilities = prediction[0][0].cpu().data.numpy().tolist()
    classes = prediction[1][0].cpu().data.numpy()
    classes = [idx_to_class[i] for i in classes]
    
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes = [cat_to_name[x] for x in classes]
    
    return probabilities, classes

def main():
    
    args = arg_parser()
    
    if(args.gpu):
        gpu = args.gpu
    else:
        print('GPU mode not specified, will use the default value - "Y"')
        gpu = "Y"
    device = device_setting(gpu)
    
    # Model loading:
    if(args.checkpoint):
        model = load_model(args.checkpoint, device)
    else:
        model = load_model('checkpoint.pth', device)

    # Image processing:
    processed_image = process_image(args.image)

    # Predictions:
    probs, classes = predict(model, processed_image, args.category_names, device, args.topk)
    print(f"Top {args.topk} predictions: {list(zip(probs, classes))}")

if __name__ == '__main__':
    main()