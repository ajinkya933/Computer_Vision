# Computer_Vision
These are the concepts for Computer Vision and Machine Learning

### Birth of RCNN

Objects of interest might have different spatial locations within the image and different aspect ratios.  
A naive approach to solve this problem would be to take different regions of interest from the image, and use a CNN to classify the presence of the object within that region.Hence, you would have to select a huge number of regions and this could computationally blow up. Therefore, algorithms like R-CNN, YOLO etc have been developed to find these occurrences and find them fast.
R-CNN
To bypass the problem of selecting a huge number of regions, Ross Girshick et al. proposed a method where we use selective search to extract just 2000 regions from the image and he called them region proposals. Therefore, now, instead of trying to classify a huge number of regions, you can just work with 2000 regions. These 2000 region proposals are generated using the selective search algorithm
These 2000 candidate region proposals are warped into a square and fed into a convolutional neural network  The CNN acts as a feature extractor. The extracted features are fed into an SVM to classify the presence of the object within that candidate region proposal.

<img width="684" alt="screen shot 2018-12-20 at 1 06 47 pm" src="https://user-images.githubusercontent.com/17012391/50270735-92522180-0458-11e9-96c3-12819c0b547f.png">

### Use of SVM

SVM is a algorithm which best seperates two classes

<img width="1312" alt="screen shot 2018-12-20 at 1 25 09 pm" src="https://user-images.githubusercontent.com/17012391/50271457-d0e8db80-045a-11e9-83fb-d1324626f6f2.png">

### Fast RCNN

We feed the input image to the CNN to generate a convolutional feature map. 

Feature map:
When there is a convolution operation between input data and a kernel the output of this operation is called a feature map:

![feature-map1](https://miro.medium.com/max/1026/1*cTEp-IvCCUYPTT0QpE3Gjg@2x.png)
![feature-map2](https://miro.medium.com/max/900/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif)


From the convolutional feature map, we identify the region of proposals(selective search algorithm is used on the feature map to identify the region proposals).  From the RoI feature vector, we use a softmax layer to predict the class of the proposed region and also the values for the bounding box. The reason “Fast R-CNN” is faster than R-CNN is because you don’t have to feed 2000 region proposals to the convolutional neural network every time. Instead, the convolution operation is done only once per image and a feature map is generated from it.

![fast-rcnn](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/rcnn.png)


### Faster R-CNN

Both of the above algorithms(R-CNN & Fast R-CNN) uses selective search to find out the region proposals. Selective search is a slow and time-consuming process affecting the performance of the network. Therefore, Shaoqing Ren et al. came up with an object detection algorithm that eliminates the selective search algorithm and lets the network learn the region proposals.

Similar to Fast R-CNN, the image is provided as an input to a convolutional network which provides a convolutional feature map. Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals.

![1 psnvmjcyqirkhdpt3cfnxa](https://user-images.githubusercontent.com/17012391/50273520-eb25b800-0460-11e9-88e2-530d990f515c.png)

### What is Relu, why is it used

### What is leaky relu


### Steps in working of OCR


### What is a activation function, why is it used ?


### What is Softmax


### What is Regularization

### overfitting

