﻿namespace DeepLearning
{
    public class NeuralNetwork
    {
        public const int NUMBER_OF_BIASES_PER_LAYER = 1;

        private Layer InputLayer { get; set; }

        private Layer[] Layers { get; set; }

        private int NumberOfNeuronsPerLayer { get; set; }

        private double[][] TrainingDataset { get; set; }

        private double[][] ExpectedResultsDataset { get; set; }

        private Func<double[], double[], double> LossFunction { get; set; }

        public NeuralNetwork(
            double[][] trainingDataset,
            double[][] expectedResultsDataset,
            int numberOfHiddenLayers,
            int numberOfNeuronsPerLayer,
            Func<double, double> activationFunctionNeurons,
            Func<double, double> activationFunctionOutputs,
            Func<double[], double[], double> lossFunction
            )
        {
            Layers = new Layer[numberOfHiddenLayers + 1];

            //Initialization first hidden layer from number of inputs
            Layers[0] = new Layer(numberOfNeuronsPerLayer, trainingDataset[0].Length, activationFunctionNeurons);

            for (int i = 1; i < numberOfHiddenLayers; i++)
            {
                Layers[i] = new Layer(numberOfNeuronsPerLayer, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER, activationFunctionNeurons);
            }

            //Initialization output layer
            Layers[numberOfHiddenLayers] = new Layer(expectedResultsDataset[0].Length, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER, activationFunctionOutputs);

            NumberOfNeuronsPerLayer = numberOfHiddenLayers;
            TrainingDataset = trainingDataset;
            ExpectedResultsDataset = expectedResultsDataset;
            LossFunction = lossFunction;
        }

        public double Test(double[] inputs, double[] expectedOutputs, Func<double[], double[], double> costFunction)
        {
            return GetError(inputs, expectedOutputs, costFunction);
        }

        public Layer Predict(double[] inputs)
        {
            InputLayer = new Layer(inputs);
            Layers[0].ComputeNeurons(InputLayer);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].ComputeNeurons(Layers[i - 1]);
            }
            return Layers[^1];
        }

        public double TrainOnData(double[] inputs, double[] expectedOutputs, Func<double[], double[], double> costFunction, double learningRate)
        {
            double[][][] weights = GetWeights();
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < NumberOfNeuronsPerLayer; j++)
                {
                    int weightIndex = 0;
                    foreach (var weight in weights[i][j]) //Variable
                    {
                        weights[i][j][weightIndex] += learningRate;
                        ModifyWeights(weights);
                        double error = GetError(inputs, expectedOutputs, costFunction);

                        weights[i][j][weightIndex] -= 2 * learningRate;
                        ModifyWeights(weights);
                        double error2 = GetError(inputs, expectedOutputs, costFunction);

                        var gradient = (error - error2) / (2 * learningRate);

                        weights[i][j][weightIndex] += learningRate - learningRate * gradient;
                        ModifyWeights(weights);
                        weightIndex++;
                    }
                }
            }
            return GetError(inputs, expectedOutputs, costFunction);
        }

        public void Train()
        {
            var trainingDataset = TrainingDataset[0..(int)(0.8f * TrainingDataset.Length)];
            var trainingExpectedResultsDataset = ExpectedResultsDataset[0..(int)(0.8f * ExpectedResultsDataset.Length)];

            var testDataset = TrainingDataset[0..(int)(0.2f * TrainingDataset.Length)];
            var testExpectedResultsDataset = ExpectedResultsDataset[0..(int)(0.2f * ExpectedResultsDataset.Length)];

            int epoch = 0;
            double lastAverageValidationLoss = 0;
            while (true)
            {
                int k = 0;
                double averageTrainingLoss = 0;
                foreach (var data in trainingDataset)
                {
                    averageTrainingLoss += TrainOnData(data, trainingExpectedResultsDataset[k++], LossFunction, 100);
                }
                averageTrainingLoss /= trainingDataset.Length;

                int l = 0;
                double averageValidationLoss = 0;
                foreach (var data in testDataset)
                {
                    averageValidationLoss += Test(data, testExpectedResultsDataset[l++], LossFunction);
                }
                averageValidationLoss /= testExpectedResultsDataset.Length;

                Console.WriteLine($"Epoch {epoch++}: train loss: {averageTrainingLoss} | validation loss : {averageValidationLoss}");
                lastAverageValidationLoss = averageTrainingLoss;

                if(lastAverageValidationLoss < averageTrainingLoss)
                {
                    Console.WriteLine($"End of training : overfitting threshold reached.");
                    break;
                }
            }
        }

        public double GetError(double[] inputs, double[] expectedOutputs, Func<double[], double[], double> costFunction)
        {
            InputLayer = new Layer(inputs);
            Layers[0].ComputeNeurons(InputLayer);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].ComputeNeurons(Layers[i - 1]);
            }
            return costFunction(Layers.Last().GetOutputs(), expectedOutputs);
        }

        public double[][][] GetWeights()
        {
            double[][][] weights = new double[Layers.Length][][];
            for (int i = 0; i < Layers.Length; i++)
            {
                weights[i] = Layers[i].GetWeights();
            }
            return weights;
        }

        public void ModifyWeights(double[][][] weights)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ModifyWeights(weights[i]);
            }
        }
    }
}
