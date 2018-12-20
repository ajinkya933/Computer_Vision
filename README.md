# Computer_Vision_Interview
These are the interview concepts for Computer Vision and Machine Learning

### Birth of RCNN

Objects of interest might have different spatial locations within the image and different aspect ratios.  
A naive approach to solve this problem would be to take different regions of interest from the image, and use a CNN to classify the presence of the object within that region.Hence, you would have to select a huge number of regions and this could computationally blow up. Therefore, algorithms like R-CNN, YOLO etc have been developed to find these occurrences and find them fast.
R-CNN
To bypass the problem of selecting a huge number of regions, Ross Girshick et al. proposed a method where we use selective search to extract just 2000 regions from the image and he called them region proposals. Therefore, now, instead of trying to classify a huge number of regions, you can just work with 2000 regions. These 2000 region proposals are generated using the selective search algorithm
These 2000 candidate region proposals are warped into a square and fed into a convolutional neural network  The CNN acts as a feature extractor. The extracted features are fed into an SVM to classify the presence of the object within that candidate region proposal.

<img width="684" alt="screen shot 2018-12-20 at 1 06 47 pm" src="https://user-images.githubusercontent.com/17012391/50270735-92522180-0458-11e9-96c3-12819c0b547f.png">

### Use of SVM


