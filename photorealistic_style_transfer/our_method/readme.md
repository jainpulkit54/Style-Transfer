To run this code, you will first need to clone the repository at: https://github.com/CSAILVision/semantic-segmentation-pytorch.git.
Then, place the provided file 'nst_segloss.py' within the cloned directory and run with the following command line arguments:

```python3 nst_segloss.py <total steps> <style weight> <content weight> <seg weight> <style image> <content image>```

total steps - total number of steps to run the image optimization for
style weight, content weight, seg weight - weighting factors that are multiplied with each of the loss terms
style image, content image - names of the style and content images

The program assumes that the style and content images are kept in a folder titled 'images' in a directory one level above.
Output images, along with loss contours are saved in a folder titled 'st_outputs'.
