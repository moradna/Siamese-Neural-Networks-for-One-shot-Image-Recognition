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
 
![image](https://user-images.githubusercontent.com/107760266/213874823-549a2eef-9923-49cb-8327-805b99797852.png)


**Experiment 2:** original paper architecture with Adam optimizer.

 ![image](https://user-images.githubusercontent.com/107760266/213874161-415eca21-cc11-412a-960b-ec289fc9ee77.png)

Results:

![image](https://user-images.githubusercontent.com/107760266/213874692-389d6ee5-6929-44c4-a7b6-0686b92cee19.png)



For both optimizers (SGD and Adam) the training time reached the maximum epochs and the time was 979s. In SGD, the loss in the validation set seems constant already after 20 epochs, but apparently the los differences between the epochs were not smaller than 0.0001, so the training did not stop. In Adam, the loss increased right at the beginning but apparently the rise was not big enough for the training to stop.

Adam optimizer gave us higher accuracy, therefor we decided to use Adam in all of the next experiments. 

The original architecture is very large, and given only 1980 pairs for training, the network learned the training set quickly, leading to overfit: the accuracy in the training set is 100%, while in the validation set it is only 68%. Moreover, the loss in the training set decreases significantly and reaches almost 0 after only 5 epochs, while the loss in the test set increases. To get over this problem we had to significantly reduce the number of parameters in each layer. 


**Experiment 3:** reduce the number of parameters in each layer.

<img width="427" alt="image" src="https://user-images.githubusercontent.com/107760266/213874229-5fe25033-86fd-46aa-9448-8795149b2267.png">

The Results:

 

![image](https://user-images.githubusercontent.com/107760266/213874268-4840e66b-ba0f-45e2-ad71-1ce7e242c85f.png)


**Experiment 4:** We added Dropout layers after each max-pooling layer and set the dropout rate to 0.4.
<img width="404" alt="image" src="https://user-images.githubusercontent.com/107760266/213874366-d2c7ab5b-d211-4887-a727-c0222e1ce279.png">


Results:

 

 
![image](https://user-images.githubusercontent.com/107760266/213874424-0a3ff530-6a35-41f9-911d-0185947fa2f8.png)


We got better results, but there is still overfitting: the loss in the training set decreases while the loss in the validation set which increases, and the accuracy in the training set increases while the accuracy in the validation remains constant. The training was stopped after 75 epochs (431s) because the model showed no improvement. To improve the overfitting we tried to increase the dropout factor. 


**Experiment 5:** We set the dropout rate to 0.7.
 
 

 
![image](https://user-images.githubusercontent.com/107760266/213874455-a4dc1f88-ea5b-4697-b492-66803201ef20.png)


This model gave us the best results: there is no significant overfitting which means the accuracy and the loss in the training set is close to the accuracy and the loss of the validation set. The training was stopped after 80 epochs (474s) because the model showed no improvement - towards the end the cost seems constant with small jumps.

## 4. Prediction Examples
4.1 Correct Classifications:

True-positive example: classify two images of the same person (class 1) for two images of the same person (class 1).

<img width="234" alt="image" src="https://user-images.githubusercontent.com/107760266/213874504-f18c3e85-b11b-4ff9-b74f-be92d07c84c3.png">


In this example there is a pair of images of the same person, that the model classified with a high confidence (89.7%) as 1. In both pictures the man has the same haircut, facial expression and suit. In addition, the background in both pictures is relatively bright. Although the man's the direction of the face was different were different in the two pictures, the model still managed to recognize that it was the same figure with a high level of confidence, probably because of the similar features we mentioned.


True-negative example: classify as two different people (class 0) for two images of the different people (class 0). 

<img width="236" alt="image" src="https://user-images.githubusercontent.com/107760266/213874529-52258357-aeeb-400b-893a-79acd85e8a60.png">


This example shows a pair of images of different people that the model classified correctly. These two people are clearly not the same person and the model indeed was able to identify this. The similar haircut can confuse the model but both their faces , their facial expression  and the background are completely different. In addition, the woman on the right is wearing glasses and a necklace and the man on the left has no glasses or necklace. All these helped the model to distinguish that these are different people.



4.2 Misclassifications: 
False-negative example: classify two images of the same person (class 1) as two different people (class 0)


<img width="230" alt="image" src="https://user-images.githubusercontent.com/107760266/213874581-72730fb8-9711-4316-85cf-258d39253919.png">


In both pictures the person's face appears in about the same angle. We assume that the misclassification in that case raised from the differences in the facial expression and the background around his face. These might have led the model to an error since it learns features from the entire image.


False-positive example: classify two different people (class 0) as the same person (class 1).

<img width="232" alt="image" src="https://user-images.githubusercontent.com/107760266/213874600-f6053930-9033-4306-91e2-42071a7ff211.png">


This example presents two men, each of them with a microphone. At first glance, the people in the pictures look similar, but it can be seen that they are different people. This similarity might have led the model to an error.
The differences in the darkness of the hair and the background are probably the reason for the lower confidence (only 69.5%).


## 5.	Conclusion
The experiments that were performed during the assignment were based on trial and error.
The results show that adding batch normalization and choosing an appropriate optimizer help improve model accuracy, while adding dropout can help reduce overfitting.
Moreover, we found out that the process of finding the best parameters can be very long and the combination of the hyperparameters set for the model impact the final results. 
The misclassification examples have shown us that in some cases the wrong classifications were caused due to some details in the background of the images that might have been misleading.
For future work we suggest using a model which will focus on facial features only, or using a different dataset with a clear background and measure the model performances on larger datasets. 
The assignment taught us new things.
First, it gave us the opportunity to build a CNN using PyTorch for the first time. Furthermore, we feel that we gained deeper understanding regarding of CNN networks and specifically Siamese networks.





