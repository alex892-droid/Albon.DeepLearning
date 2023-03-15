# DeepLearning

This project has been made to make implementation of neural network in C# easy. 

Here's the step to implement a neural network (at the time only feed-forward neural network) with this library :

[1] To create a new neural network, you just need to call its constructor and feed it the parameters :
  1. Training data (must be array of inputs (itself an array of double) double[][])
  2. Expected Results Data (must be array of outputs (itself an array of double) double[][])
  3. Number of hidden layers
  4. Number of neurons per layer
  5. Activation function (find some on the MathTools static class that comes with the project)
  6. Activation function for outputs (find some on the MathTools static class that comes with the project)
  7. Loss function (find some on the MathTools static class that comes with the project)
  8. Training rate (must be double)
  
[2] Call Train() method of the neural network created in 1) to train the neural network.

[3] Call Predict() method of the neural network to predict, it takes an input double[] and return an output double[] of the prediction.
  
