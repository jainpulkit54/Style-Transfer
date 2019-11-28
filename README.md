# Style-Transfer

The repository has three folders namely:
1) classifier
2) fast_style_transfer
3) images

# Classifier
The classifier contains the "training.py", "testing.py" files and also the "resnet18parameters.pth" file.
The dataset used to train the model happens to be "Places 365" dataset.

## Places 365 Dataset
The dataset can be downloaded using the following link:
http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar

Four classes out of 365 were used for training our classifier. The four classes are:
1) desert_sand
2) hot_spring
3) ocean
4) skyscraper

# Fast Style Transfer
This contains two folders namely:
1) image_generation_code
2) training_code

Use the "training.py" file present in "training_code" folder to train a model for a particular style. Other instructions are mentioned in the .py file itself.
Use the "image_generation.py" file present in "image_generation_code" folder to genrate images specifying a model for a particular style. Other instructions are mentioned in the .py file itself.

# Images
This folder contain the images we generated.

# References
[1] L. A. Gatys, A. S. Ecker, and M. Bethge, “Image style transfer using convolutional neural networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2414–2423, 2016.
[2] F. Luan, S. Paris, E. Shechtman, and K. Bala, “Deep photo style transfer,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 4990–4998, 2017.
[3] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style transfer and super-resolution,” in European conference on computer vision, pp. 694–711, Springer, 2016.
