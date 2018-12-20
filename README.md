# Computer_Vision_Interview
These are the interview questions for Computer Vision and Machine Learning

Objects of interest might have different spatial locations within the image and different aspect ratios.  
A naive approach to solve this problem would be to take different regions of interest from the image, and use a CNN to classify the presence of the object within that region.Hence, you would have to select a huge number of regions and this could computationally blow up. Therefore, algorithms like R-CNN, YOLO etc have been developed to find these occurrences and find them fast.
R-CNN
To bypass the problem of selecting a huge number of regions, Ross Girshick et al. proposed a method where we use selective search to extract just 2000 regions from the image and he called them region proposals. Therefore, now, instead of trying to classify a huge number of regions, you can just work with 2000 regions. These 2000 region proposals are generated using the selective search algorithm
These 2000 candidate region proposals are warped into a square and fed into a convolutional neural network that produces a 4096-dimensional feature vector as output
