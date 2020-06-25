# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

Run the following to train the network:
```python train.py --architecture 'vgg16' --learning_rate 0.001 --dropout 0.05 --epochs 10 --gpu  'Y'```

Run the following in terminal to get the prediction results:
```python predict.py --image 'flowers/test/1/image_06743.jpg' --topk 5 --category_names 'cat_to_name.json' --checkpoint 'checkpoint.pth'```