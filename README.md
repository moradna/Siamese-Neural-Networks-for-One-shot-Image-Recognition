# Siamese-Neural-Networks-for-One-shot-Image-Recognition

**Assignment purpose**: Use convolutional neural networks (CNN) to carry out the task of facial recognition. We implemented a one-shot classification solution using Siamese Neural Networks based on the paper “Siamese Neural Networks for One-shot Image Recognition” (Koch et al. 2015).
## 1. Data Description

The” Labeled Faces in the Wild-a” image collection is a database of labeled, face images intended for studying Face Recognition in unconstrained images.
In the assignment we were given a specific train/test split of the pairs of images (via text files) so that no image is shared by both sets. The train set contain a total of 2200 image pairs for training of which 1100 which positive pairs and 1100 negative pairs - where negative and positive refer to whether the figure appear in the image is the same. For testing we were provided with a separate list of 1000 pairs, 500 positive and 500 negative. There is no overlap of peoples’ images between the train set and the test set (except two characters appear in both sets with different images). This fact ensures that the learning task that the model is dealing with is one-shot learning task. The largest number of appearances for the same character was 6. 

For training purposes, we chose to split the original train set into train set and validation set in ratio of 90:10 respectively. As a result, we got a train set of 1980 pairs of images and a validation set of 220 images. We decided on this ratio since the original train set is small and we wanted to keep as many pairs as possible for the training of our model and not “waste” them on the validation set.

## 2. Model architecture:
As described in the paper, we followed the architecture presented in the next illustration:

<img width="452" alt="image" src="https://user-images.githubusercontent.com/107760266/213873864-adf8647b-f4d9-4dd5-a306-3b881d0a5062.png">



After testing on the validation set we choose the following configuration:

-	Batch size: In our trial-and-error phase, we tried 32, 64, and 128 batch sizes, but 32 produced the best results.
-	Weight initialization: for all edges was done as described in the paper - a normal distribution with a mean of 0 and a standard deviation of 0.01
-	BatchNorm: Using BatchNorm before each Max polling helped to get better results in all our experiments.
-	Max epochs for the model: 100
-	Loss function: Binary Cross Entropy by torch.nn because the model classification is 0 or 1.
-	Early stopping criteria: the training stops in a case in which the loss of the validation set does not show improvement for 5 epochs.
-	Learning rate: We are decreasing the learning rate factor by 1% after each epoch as described in the paper, using lr_scheduler function.

## 3.	Experimental Setup

**Experiment 1:** original paper architecture with SGD optimizer.

<img width="352" alt="image" src="https://user-images.githubusercontent.com/107760266/213874015-92383360-131a-467a-ada3-9a47d12afbcb.png">

Results: 


<img width="486" alt="image" src="https://user-images.githubusercontent.com/107760266/213874048-076862eb-cc29-4c77-bcd3-f39a06b959ff.png">

<img width="484" alt="image" src="https://user-images.githubusercontent.com/107760266/213874055-d244e974-32df-4ac3-9bcd-377919a02595.png">

<img width="277" alt="image" src="https://user-images.githubusercontent.com/107760266/213874068-aa81c23c-f3f9-44b6-91fa-6f5fae52ec6d.png">


**Experiment 2:** original paper architecture with Adam optimizer.

 ![image](https://user-images.githubusercontent.com/107760266/213874161-415eca21-cc11-412a-960b-ec289fc9ee77.png)


<img width="347" alt="image" src="https://user-images.githubusercontent.com/107760266/213874169-77c46e0c-7214-4f86-b736-23c89cbd5061.png">



For both optimizers (SGD and Adam) the training time reached the maximum epochs and the time was 979s. In SGD, the loss in the validation set seems constant already after 20 epochs, but apparently the los differences between the epochs were not smaller than 0.0001, so the training did not stop. In Adam, the loss increased right at the beginning but apparently the rise was not big enough for the training to stop.

Adam optimizer gave us higher accuracy, therefor we decided to use Adam in all of the next experiments. 

The original architecture is very large, and given only 1980 pairs for training, the network learned the training set quickly, leading to overfit: the accuracy in the training set is 100%, while in the validation set it is only 68%. Moreover, the loss in the training set decreases significantly and reaches almost 0 after only 5 epochs, while the loss in the test set increases. To get over this problem we had to significantly reduce the number of parameters in each layer. 


**Experiment 3:** reduce the number of parameters in each layer.

<img width="427" alt="image" src="https://user-images.githubusercontent.com/107760266/213874229-5fe25033-86fd-46aa-9448-8795149b2267.png">

The Results:

<img width="515" alt="image" src="https://user-images.githubusercontent.com/107760266/213874252-1ab3c3df-a2dd-421c-ac1a-9a930907523a.png">

 

![image](https://user-images.githubusercontent.com/107760266/213874268-4840e66b-ba0f-45e2-ad71-1ce7e242c85f.png)

