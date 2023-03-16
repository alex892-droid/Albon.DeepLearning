using DeepLearning;

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
    0.0001,
    0.01);


neuralNetwork.Train();


var result = neuralNetwork.Predict(new double[4] { 1, 2, 3, 4 });
Console.WriteLine($"{result[0]} : {(5 * 1 + 2 * 2 + 1.5 * 3 + 4 + 5)}");