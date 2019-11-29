# Style-Transfer and Data augmentation
This repository contains the codes from our work on a review of existing methods of neural style transfer and their application to data augmentation for a classification task.

## Authors:
<a href="https://github.com/jainpulkit54">Pulkit jain</a> </br>
<a href="https://github.com/rohitma38">Rohit M A</a> </br>
<a href="https://github.com/shyama95">Shyama P</a> </br>

The repository has the following folders:
1) classifier - implementation of the classification task
2) fast_style_transfer - implementation of fast style transfer for data augmentation
3) images
4) photorealistic_style_transfer - an existing method and an implementation of our own

## Classifier
The classifier contains the "training.py", "testing.py" files and also the "resnet18parameters.pth" file.
The dataset used to train the model happens to be the "Places 365" dataset.

### Places 365 Dataset
The dataset can be downloaded using the following link:
http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

Four classes out of 365 were used for training our classifier. The four classes are:
1) desert_sand
2) hot_spring
3) ocean
4) skyscraper

## Fast Style Transfer
This contains two folders namely:
1) image_generation_code
2) training_code

Use the "training.py" file present in "training_code" folder to train a model for a particular style. Other instructions are mentioned in the .py file itself.
Use the "image_generation.py" file present in "image_generation_code" folder to genrate images specifying a model for a particular style. Other instructions are mentioned in the .py file itself.

## Images
This folder contain the images we generated.

## Photo-realistic style transfer
This contains two folders namely:
1) our_method - a modification of the method in [1], with an added segmentation loss term based on segmentation maps obtained using the network from [4]
2) deep_photo - implementation of the method in [2]


## References
[1] L. A. Gatys, A. S. Ecker, and M. Bethge, “Image style transfer using convolutional neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2414–2423, 2016.
[2] F. Luan, S. Paris, E. Shechtman, and K. Bala, “Deep photo style transfer,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4990–4998, 2017.  
[3] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in European conference on computer vision, pp. 694–711, Springer, 2016.  
[4] Zhou, Bolei, et al. "Semantic understanding of scenes through the ade20k dataset." International Journal of Computer Vision 127.3, pp. 302-321, 2019.
