using DeepLearning;
/*
double[][] dataset = new double[1000][];
double[][] datasetResults = new double[1000][];
for (int i = 0; i < 1000; i++)
{
    double[] variables = new double[4];
    variables[0] = MathTools.Random.Next(0, 10);
    variables[1] = MathTools.Random.Next(0, 10);
    variables[2] = MathTools.Random.Next(0, 10);
    variables[3] = MathTools.Random.Next(0, 10);
    dataset[i] = variables;
    datasetResults[i] = new double[1] { 5 * variables[0] + 2 * variables[1] + 1.5 * variables[2] + variables[3] + 5 };
}

NeuralNetwork neuralNetwork = new NeuralNetwork(
    dataset,
    datasetResults,
    2,
    9,
    MathTools.ReLU,
    MathTools.ReLU,
    MathTools.MeanSquaredError,
    0.0001);
neuralNetwork.Train();

var result = neuralNetwork.Predict(new double[4] { 1, 2, 3, 4 });
Console.WriteLine($"{result.GetOutputs()[0]} : {(5 * 1 + 2 * 2 + 1.5 * 3 + 4 + 5)}");*/

double[][] results = new double[500][];
double[][] trainingData = new double[500][];
for (int i = 0; i < 500; i++)
{
    results[i] = new double[1];
    results[i][0] = 1;

    trainingData[i] = new double[5];
    trainingData[i][0] = 0.01;
    trainingData[i][1] = 0.01;
    trainingData[i][2] = 0.01;
    trainingData[i][3] = 0.01;
    trainingData[i][4] = 0.01;
}

NeuralNetwork neuralNetwork = new NeuralNetwork(trainingData, results, 1, 5, MathTools.ReLU, MathTools.Sigmoid, MathTools.BinaryCrossEntropyLoss, 0.001);
neuralNetwork.Train();