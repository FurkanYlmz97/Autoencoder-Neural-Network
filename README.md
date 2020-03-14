# Autoencoder-Neural-Network
In this repository I have implemented an autoencoder neural network with a single hidden layer for unsupervised feature extraction from natural images.

For this sake the following cost function have been minimized:
![Cost Function](https://user-images.githubusercontent.com/48417171/76660810-d7490300-658a-11ea-8269-7abc831e79a8.png)

**Preprocessing**

Before starting to building the network, I have loaded the data which contains 16x16 patches extracted from various natural images. The goal was converting these RGB colored images to one band gray image. To do that first I have taken the three bands of the image and multiplying with come weights and adding them to each other I have achieved one band image. For this purpose, the used function is 𝑌=0.2126∗𝑅+0.7152∗𝐺+0.0722∗𝐵. To normalize the one band image data, I have first made the mean zero by removing the mean value from every pixel. The next step was clipping where I have clipped the data with ±3 standard deviation. Finally, to prevent saturation I have map the clipped data to the range [0.1 0.9]. After preprocessing is done, I have randomly selected 10 images and plotted the preprocessed and untouched version of them as following:


![1](https://user-images.githubusercontent.com/48417171/76690729-3292f800-6654-11ea-9918-7c7830b48404.png)


From the figures 1 to 20 it can be seen that preprocessed images (Gray images) have the almost same shape with the RGB images. The reason behind this is that high valued pixels in red, green or blue bands in the RGB image result in high-value pixel values in the gray images. However, the three bands do not equally contribute to the gray image because of the function, 𝑌=0.2126∗𝑅+0.7152∗𝐺+0.0722∗𝐵. Thus, I expect that the gray images will be more look like red band images. The lowest contribution comes from the green band.

**Training**

Before starting the training process, I have initialized the weights with the uniformly in range [𝑤𝑜,−𝑤𝑜] where 𝑤𝑜=(6/(𝐿𝑝𝑟𝑒+𝐿𝑝𝑜𝑠𝑡))^(1/2) for our network 𝐿𝑝𝑟𝑒+𝐿𝑝𝑜𝑠𝑡=256+𝐻𝑖𝑑𝑑𝑒𝑛 𝑁𝑒𝑢𝑟𝑜𝑛 𝑁𝑢𝑚𝑏𝑒𝑟, where the input and the output layer always have 256 neurons. For this section, I have taken the hidden neuron number equal to 64. And the following cost function has been used.

![Cost Function](https://user-images.githubusercontent.com/48417171/76660810-d7490300-658a-11ea-8269-7abc831e79a8.png)


For this part 𝜆=5∗10^(-4) , and also𝜌𝑏 is the average activation of hidden units. I have trained this network by using batch gradient descent and not solving them with gradient solver because taking the gradient of this function is easy and I couldn’t find any usable gradient solver for python. The gradient of this function has been taken as follows,


![2](https://user-images.githubusercontent.com/48417171/76690777-c4026a00-6654-11ea-9f48-78425167b056.png)

I have trained my NN with different parameters and finally, I have found that the output results look best for the parameters, 𝛽=0.2,𝜌=0.5 the output and input of this NN is as following:


![3](https://user-images.githubusercontent.com/48417171/76690797-0a57c900-6655-11ea-90d8-11253608cb4a.png)


Note that in this part it is only allowed to change the parameters 𝛽 𝑎𝑛𝑑 𝜌. First, I have searched what does it mean changing values of 𝜌, I have learned that in KL divergence the error increases if the two given parameters are far away, i.e. the difference between the parameters is big and KL divergence equals 0 when the two parameters are equal. Therefore, by considering that in autoencoders we are expecting an output that is similar to the input. So, I have toughed that since the input’s mean is close to 0.5 (remember that I have mapped it in the range (0.1, 0.9)) and since KL divergences gives the mean of activation of the hidden units i.e. output without the weights and assuming that multiplying with weights will not dramatically change the mean of the hidden activations. The 𝜌 value equal to 0.5 will give a good result where the input also has 0.5 mean and the output should be penalized if the mean of it gets far from 0.5. For this network
Furthermore, 𝛽 value determines how much should we give importance that the mean of the output image should close to the 𝜌 value. Increasing this term much will make our NN a worse autoencoder since the other penalizing terms affect will decrease. Therefore, first, I have done several tests for changing 𝜌 here is the result from 𝛽=0.1,𝜌=0.2, where I have taken a learning rate as 0.05 with some trials without changing the two parameters. After I have found the best learning rate equal to 0.05, I plotted the outputs as follows,


![4](https://user-images.githubusercontent.com/48417171/76690798-0af05f80-6655-11ea-904a-685b5e9c880d.png)


From the figure above it can be seen that this value of 𝜌 also gives reasonable outputs but in this example I did not like the output 8 where the output is close to the input but there is another shape i.e. the parallel dark lines do not match. I thought maybe I can do better with increasing the 𝜌 value. So, I have increased the 𝜌 value and trained the NN again. This is the result of 𝜌=0.5 with same 𝛽 value.


![5](https://user-images.githubusercontent.com/48417171/76690799-0b88f600-6655-11ea-8b68-8dafba1f2435.png)

I have found the outputs in the figure above looks similar but his time there are some pixel value differences. For example, in output 10 the shape is similar but the output is not clear enough. Thus, I have increased 𝛽 value for this one and the result is in figure 22. These parameters have given the best output in my opinion. Also, increasing 𝛽 further to much higher values resulted in divergence and saturation while training. I have tried one more with 𝛽=0.3. The result is as follows,


![6](https://user-images.githubusercontent.com/48417171/76690800-0c218c80-6655-11ea-9b75-4d513c19d97c.png)


Again, the result is satisfying in my opinion. However, I would prefer the one with values 𝛽=0.3,𝜌=0.5 where in this one output shapes look more different if I compare it with NN with 𝛽=0.2,𝜌=0.5 parameters. Especially the small shape changes in outputs 7 and 6 made me to choose the other NN.
Furthermore, I am conducting a performance rating by looking at the qualities of the output images. This can be tricky and not the best way to see whether the two images look like similar to each other or not. However, since in autoencoders we are not expecting to have an %100 similarity and we are just trying to learn the features from the inputs, using some parameters for testing the similarities between the input and outputs will be not reasonable a lot. Also, it is important to notice that I am doing an full batch training and showing the training results, therefore it is not certain that this network can perform well in terms of generalizing but in this part of the assignment it is not wanted from us to train an non-overfitted NN and by comparing the quality of the outputs I believe and assume that the best parameters for my NN are 𝛽=0.2,𝜌=0.5.
