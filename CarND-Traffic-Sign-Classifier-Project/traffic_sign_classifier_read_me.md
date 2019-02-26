## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

[//]: # (Image References)

[image1]: ./output_folder/29745.jpg  "traning_image_1"
[image2]: ./output_folder/28215.jpg  "traning_image_2"
[image3]: ./output_folder/18774.jpg  "traning_image_3"
[image4]: ./output_folder/10020.jpg  "traning_image_4"
[image5]: ./output_folder/17239.jpg  "traning_image_5"
[image6]: ./output_folder/8214.jpg  "traning_image_6"
[image7]:  ./output_folder/21082.jpg  "traning_image_7"
[image8]:  ./output_folder/30860.jpg  "traning_image_8"
[image9]:  ./output_folder/12023.jpg  "traning_image_9"
[image10]: ./output_folder/20034.jpg  "traning_image_10"
[image11]: ./output_folder/histo.jpg  "histogram"

[image12]: ./german_image/img_1.jpg  "German_sign_1"
[image13]: ./german_image/img_2.jpg  "German_sign_2"
[image14]: ./german_image/img_3.jpg  "German_sign_3"
[image15]: ./german_image/img_4.jpg  "German_sign_4"
[image16]: ./german_image/img_5.jpg  "German_sign_5"
[image17]: ./german_image/img_6.jpg  "German_sign_6"



### Load the data set

In this project we need to train a deep learning neural network to classify the German Traffic Sign Recognition Benchmark dataset.
There are about 40 different types of German traffic signs in the dataset, each image is of size 32x32 pixels.

Import the necessary libraries. The datafile is a pickle contaning traning,test and validation file.

After loading the files from the pickle, we create X_train, y_train ,X_valid, y_valid,X_test, y_test.

#### code snippet:

import pickle

training_file = "traffic_sign_data/train.p"
validation_file="traffic_sign_data/test.p"
testing_file = "traffic_sign_data/valid.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Explore, summarize and visualize the data set

Using numpy we calculate summary statistics of the traffic signs data set.

#### code snippet:


n_train = len(X_train)
n_validation = len(X_valid)
n_test = len(X_test)
image_shape = X_train[0].shape
n_classes = len(np.unique(y_train))

Below is the summary statistics:

Number of training examples = 34799
Number of testing examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43

Here is an exploratory visualization of the data set. We randomly extract 10 images from the traning set and display the images.


#### code snippet:

for i in range(10):
    index = random.randint(0, len(X_train))
    print(index)
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])
    cv2.imwrite("./output_folder/"+str(index)+".jpg",image)


![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]

#### code snippet:

Below is the histogram representing the number of classes and their distributions.

plt.figure(figsize=(10,10))
plt.hist(y_train, bins = n_classes)
plt.xlabel('Class',fontsize=20)
plt.ylabel('Number of training examples',fontsize=20)
plt.grid()
plt.savefig("./output_folder/histo.jpg")
plt.show()

![alt text][image11]

### Design, train and test a model architecture

The model architecture is based on the LeNet model architecture. 

Layer                           Description

Input                 32x32x1 gray scale image

Convolution           5x5 1x1 stride, valid padding, outputs 28x28x6

RELU 

Max pooling	          2x2 stride, outputs 14x14x6

Convolution           5x5 1x1 stride, valid padding, outputs 10x10x16

RELU 

Max pooling           2x2 stride, outputs 5x5x16

Flatten               outputs 400

Dropout

Fully connected       outputs 120

RELU 

Dropout

Fully connected       outputs 84

RELU

Dropout

Fully connected       outputs 43


Initially the LeNet model is trained and we get traning accuracy to be 100% so inorder to prevent overfitting the LeNet model is modified.

In the modified version we added the droupots which prevents from overfitting of he model.

The model was trained with Adam optimizer and the following hyperparameters were chosen after trail analysis:

batch size: 128

number of epochs: 140

learning rate: 0.0009

Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1

keep probalbility of the dropout layer: 0.5

Final result for the above model specification are:

validation set accuracy of 94.9%

test set accuracy of 97.3%

### Use the model to make predictions on new images

The trained model was used to predict the trafic sign for the 6 different traffic signals. 

Below are the images

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

#### code : 

new_images_label = [2, 34, 17, 14, 4, 13]


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))new_images_normalized = preProcess(new_images)
    results = np.argmax(np.array(sess.run(logits, feed_dict={x: new_images_normalized,keep_prob: 1.0})),axis = 1)
    
Predictions :

02 ------->Prediction 02
34 ------->Prediction 12
17 ------->Prediction 17
14 ------->Prediction 14
04 ------->Prediction 04
13 ------->Prediction 13
Accuracy ------->0.83

In the above prediction we see there is one misclassification which causes our test accuracy to go down.

### Analyze the softmax probabilities of the new images

The softmax function codeand output below:

#### code snippet : 

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    softmax = sess.run(tf.nn.softmax(logits), feed_dict = {x:new_images_normalized,keep_prob: 1.0})
    top5 = sess.run(tf.nn.top_k(softmax, k=5))
    
INFO:tensorflow:Restoring parameters from ./lenet
Top 5 Predictions for each image :
 [[ 2  1  5  3 13]
 [12 40 35 33  1]
 [17 14 34 38 12]
 [14 17 34 38  0]
 [ 4  0  1 14  8]
 [13  2 35 12 34]]
Top 5 corresponding probabilities:
 [[  1.00000000e+00   2.07788986e-13   2.08285865e-16   5.89839044e-18
    4.09250183e-23]
 [  7.35332072e-01   1.00768484e-01   4.69377339e-02   3.48111726e-02
    1.68776792e-02]
 [  9.59680974e-01   9.91247501e-03   8.28970596e-03   7.83945806e-03
    4.74834861e-03]
 [  1.00000000e+00   3.70904132e-12   1.77533796e-13   8.57074015e-14
    7.85386025e-14]
 [  7.28913307e-01   2.66598046e-01   4.47354186e-03   1.23373229e-05
    2.30704313e-06]
 [  1.00000000e+00   3.92448281e-20   3.19586682e-21   1.08710667e-21
    5.41269804e-22]]


### References:

https://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent

https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc

https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

https://navoshta.com/traffic-signs-classification/