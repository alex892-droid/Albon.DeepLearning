namespace DeepLearning
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

        private double LearningRate { get; set; }

        public NeuralNetwork(
            double[][] trainingDataset,
            double[][] expectedResultsDataset,
            int numberOfHiddenLayers,
            int numberOfNeuronsPerLayer,
            Func<double, double> activationFunctionNeurons,
            Func<double, double> activationFunctionOutputs,
            Func<double[], double[], double> lossFunction,
            double learningRate
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
            LearningRate = learningRate;
        }

        public double Test(double[] inputs, double[] expectedOutputs)
        {
            return GetError(inputs, expectedOutputs);
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

        public double TrainOnData(double[] inputs, double[] expectedOutputs)
        {
            double[][][] weights = GetWeights();
            for (int i = 0; i < Layers.Length; i++)
            {
                for (int j = 0; j < NumberOfNeuronsPerLayer; j++)
                {
                    int weightIndex = 0;
                    foreach (var weight in weights[i][j]) //Variable
                    {
                        weights[i][j][weightIndex] += LearningRate;
                        ModifyWeights(weights);
                        double error = GetError(inputs, expectedOutputs);

                        weights[i][j][weightIndex] -= 2 * LearningRate;
                        ModifyWeights(weights);
                        double error2 = GetError(inputs, expectedOutputs);

                        var gradient = (error - error2) / (2 * LearningRate);

                        weights[i][j][weightIndex] += LearningRate - LearningRate * gradient;
                        ModifyWeights(weights);
                        weightIndex++;
                    }
                }
            }
            return GetError(inputs, expectedOutputs);
        }

        public void Train()
        {
            var trainingDataset = TrainingDataset[0..(int)(0.8f * TrainingDataset.Length)];
            var trainingExpectedResultsDataset = ExpectedResultsDataset[0..(int)(0.8f * ExpectedResultsDataset.Length)];

            var testDataset = TrainingDataset[0..(int)(0.2f * TrainingDataset.Length)];
            var testExpectedResultsDataset = ExpectedResultsDataset[0..(int)(0.2f * ExpectedResultsDataset.Length)];

            int epoch = 0;
            double lastAverageValidationLoss = double.MaxValue;
            while (true)
            {
                double averageTrainingLoss = 0;
                int k = 0;
                foreach (var data in trainingDataset)
                {
                    averageTrainingLoss += TrainOnData(data, trainingExpectedResultsDataset[k++]);
                }
                averageTrainingLoss /= trainingDataset.Length;

                int l = 0;
                double averageValidationLoss = 0;
                foreach (var data in testDataset)
                {
                    averageValidationLoss += Test(data, testExpectedResultsDataset[l++]);
                }
                averageValidationLoss /= testExpectedResultsDataset.Length;

                Console.WriteLine($"Epoch {epoch++}: train loss: {averageTrainingLoss} | validation loss : {averageValidationLoss}");

                if(lastAverageValidationLoss < averageValidationLoss)
                {
                    Console.WriteLine($"End of training : overfitting threshold reached.");
                    break;
                }

                if (lastAverageValidationLoss == averageValidationLoss)
                {
                    Console.WriteLine($"End of training : learning ineffective. Increase number of neurons or increase learning rate.");
                    break;
                }

                if (averageValidationLoss == double.NaN)
                {
                    Console.WriteLine($"End of training : Error too high to be calculated. Adjust parameters.");
                    break;
                }

                lastAverageValidationLoss = averageValidationLoss;
            }
        }

        public double GetError(double[] inputs, double[] expectedOutputs)
        {
            InputLayer = new Layer(inputs);
            Layers[0].ComputeNeurons(InputLayer);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].ComputeNeurons(Layers[i - 1]);
            }
            return LossFunction(Layers.Last().GetOutputs(), expectedOutputs);
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
