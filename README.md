
# Computer_Vision
These are the concepts for Computer Vision and Machine Learning

# Fundamental structure of NN

<img width="500" alt="NN" src="https://miro.medium.com/max/1401/1*uulvWMFJMidBfbH9tMVNTw@2x.png">

### Birth of RCNN

Objects of interest might have different spatial locations within the image and different aspect ratios.  
A naive approach to solve this problem would be to take different regions of interest from the image, and use a CNN to classify the presence of the object within that region.Hence, you would have to select a huge number of regions and this could computationally blow up. Therefore, algorithms like R-CNN, YOLO etc have been developed to find these occurrences and find them fast.
R-CNN
To bypass the problem of selecting a huge number of regions, Ross Girshick et al. proposed a method where we use selective search to extract just 2000 regions from the image and he called them region proposals. Therefore, now, instead of trying to classify a huge number of regions, you can just work with 2000 regions. These 2000 region proposals are generated using the selective search algorithm
These 2000 candidate region proposals are warped into a square and fed into a convolutional neural network  The CNN acts as a feature extractor. The extracted features are fed into an SVM to classify the presence of the object within that candidate region proposal.

<img width="500" alt="screen shot 2018-12-20 at 1 06 47 pm" src="https://user-images.githubusercontent.com/17012391/50270735-92522180-0458-11e9-96c3-12819c0b547f.png">

### Use of SVM

SVM is a algorithm which best seperates two classes

<img width="500" alt="screen shot 2018-12-20 at 1 25 09 pm" src="https://user-images.githubusercontent.com/17012391/50271457-d0e8db80-045a-11e9-83fb-d1324626f6f2.png">

### Fast RCNN

We feed the input image to the CNN to generate a convolutional feature map. 

Feature map:
When there is a convolution operation between input data and a kernel the output of this operation is called a feature map:

<img width="400" alt="featuremap1" src="https://miro.medium.com/max/1026/1*cTEp-IvCCUYPTT0QpE3Gjg@2x.png">

<img width="500" alt="featuremap2" src="https://miro.medium.com/max/900/1*VVvdh-BUKFh2pwDD0kPeRA@2x.gif">


From the convolutional feature map, we identify the region of proposals(selective search algorithm is used on the feature map to identify the region proposals).  From the RoI feature vector, we use a softmax layer to predict the class of the proposed region and also the values for the bounding box. The reason “Fast R-CNN” is faster than R-CNN is because you don’t have to feed 2000 region proposals to the convolutional neural network every time. Instead, the convolution operation is done only once per image and a feature map is generated from it.

<img width="500" alt="featuremap2" src="https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2018/10/Fast-rcnn.png">

### Faster R-CNN

Both of the above algorithms(R-CNN & Fast R-CNN) uses selective search to find out the region proposals. Selective search is a slow and time-consuming process affecting the performance of the network. Therefore, Shaoqing Ren et al. came up with an object detection algorithm that eliminates the selective search algorithm and lets the network learn the region proposals.

Similar to Fast R-CNN, the image is provided as an input to a convolutional network which provides a convolutional feature map. Instead of using selective search algorithm on the feature map to identify the region proposals, a separate network is used to predict the region proposals.

<img width="500" alt="fasterrcnn" src="https://user-images.githubusercontent.com/17012391/50273520-eb25b800-0460-11e9-88e2-530d990f515c.png">

 have summarized below the steps followed by a Faster R-CNN algorithm to detect objects in an image:

Take an input image and pass it to the ConvNet which returns feature maps for the image
Apply Region Proposal Network (RPN) on these feature maps and get object proposals
Apply ROI pooling layer to bring down all the proposals to the same size
Finally, pass these proposals to a fully connected layer in order to classify any predict the bounding boxes for the image

###  Stride and Padding
Stride specifies how much we move the convolution filter at each step. By default the value is 1, as you can see in the figure below.
<img width="500" alt="featuremap2" src="https://miro.medium.com/max/790/1*L4T6IXRalWoseBncjRr4wQ@2x.gif">

We see that the size of the feature map is smaller than the input, because the convolution filter needs to be contained in the input. If we want to maintain the same dimensionality, we can use padding to surround the input with zeros. Check the animation below.

<img width="500" alt="featuremap2" src="https://miro.medium.com/max/1063/1*W2D564Gkad9lj3_6t9I2PA@2x.gif">

### Pooling
Contrary to the convolution operation, pooling has no parameters. It slides a window over its input, and simply takes the max value in the window. Similar to a convolution, we specify the window size and stride.

<img width="500" alt="featuremap2" src="https://miro.medium.com/max/1172/1*ReZNSf_Yr7Q1nqegGirsMQ@2x.png">

### What is Relu, why is it used
The ReLU (Rectified Linear Unit) is the most used activation function in the world right now
Range: [ 0 to infinity)

<img width="500" alt="featuremap2" src="https://miro.medium.com/max/726/1*XxxiA0jJvPrHEJHD4z893g.png">

### Why is relu used?
Relu is also called as a rectifier, The purpose of applying the rectifier function is to increase the non-linearity in our images
The reason we want to do that is that images are naturally non-linear.When you look at any image, you'll find it contains a lot of non-linear features (e.g. the transition between pixels, the borders, the colors, etc.).
 
 <img width="500" alt="relu" src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_1.png">
 
The rectifier serves to break up the linearity even further. The input image
This black and white image is the original input image.

<img width="500" alt="relu" src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_2.png">

Rectification
What the rectifier function does to an image like this is remove all the black elements from it, keeping only those carrying a positive value (the grey and white colors).

<img width="500" alt="relu" src="https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/71_blog_image_4.png">

### What is leaky relu
The issue with relu is that all the negative values become zero immediately which decreases the ability of the model to fit or train from the data properly. That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph.

<img width="500" alt="relu" src="https://miro.medium.com/max/704/1*A_Bzn0CjUgOXtPCJKnKLqA.jpeg">
The range of the Leaky ReLU is (-infinity to infinity).

### Steps in working of OCR


### What is a activation function, why is it used ?
It’s just a  function where you pass in data to get an output. It is also known as Transfer Function.
It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc. (depending upon the function).
The Activation Functions can be basically divided into 2 types-

Linear Activation Function

Non-linear Activation Functions


### What is Softmax
Softmax classifiers give you probabilities for each class

### What is Regularization

For any machine learning problem, essentially, you can break your data points into two components — pattern + stochastic noise. For instance, if you were to model the price of an apartment, you know that the price depends on the area of the apartment, no. of bedrooms, etc. So those factors contribute to the pattern — more bedrooms would typically lead to higher prices. However, all apartments with the same area and no. of bedrooms do not have the exact same price. The variation in price is the noise.

Now the goal of machine learning is to model the pattern and ignore the noise. Anytime an algorithm is trying to fit the noise >> in addition to the pattern, it is overfitting.

### Two most important regularization techniques in machine learning are: Dropout and Batch Normalization

### Dropout
We are disabling neurons on purpose and the network actually performs better. The reason is that dropout prevents the network to be too dependent on a small number of neurons, and forces every neuron to be able to operate independently. 
<img width="500" alt="dropout" src="https://miro.medium.com/max/914/1*7LrJUUXIO8ewrbuUIbUkXQ@2x.png">

### overfitting

References: https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-step-1b-relu-layer/
