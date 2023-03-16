using System.Xml.Schema;

namespace DeepLearning
{
    public class NeuralNetwork
    {
        private const int NUMBER_OF_BIASES_PER_LAYER = 1;

        private Layer InputLayer { get; set; }

        private Layer[] Layers { get; set; }

        private double[][] TrainingDataset { get; set; }

        private double[][] ExpectedResultsDataset { get; set; }

        private double MinLearningPrecision { get; set; }

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
            double learningRate,
            double minLearningPrecision = double.MinValue
            )
        {
            if(trainingDataset.Length < 100)
            {
                throw new ArgumentException("Not enough data to train a neural network.");
            }

            Layers = new Layer[numberOfHiddenLayers + 1];

            //Initialization first hidden layer from number of inputs
            Layers[0] = new Layer(numberOfNeuronsPerLayer, trainingDataset[0].Length + NUMBER_OF_BIASES_PER_LAYER, activationFunctionNeurons);

            for (int i = 1; i < numberOfHiddenLayers; i++)
            {
                Layers[i] = new Layer(numberOfNeuronsPerLayer, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER, activationFunctionNeurons);
            }

            //Initialization output layer
            Layers[numberOfHiddenLayers] = new Layer(expectedResultsDataset[0].Length, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER, activationFunctionOutputs);

            TrainingDataset = trainingDataset;
            ExpectedResultsDataset = expectedResultsDataset;
            LossFunction = lossFunction;
            LearningRate = learningRate;
            MinLearningPrecision = minLearningPrecision;
        }

        #region Public Methods

        public double[] Predict(double[] inputs)
        {
            InputLayer = new Layer(inputs);
            Layers[0].ComputeNeurons(InputLayer);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].ComputeNeurons(Layers[i - 1]);
            }
            return Layers[^1].GetOutputs();
        }

        public double Train()
        {
            var trainingDataset = TrainingDataset[0..(int)(0.8f * TrainingDataset.Length)];
            var trainingExpectedResultsDataset = ExpectedResultsDataset[0..(int)(0.8f * ExpectedResultsDataset.Length)];

            var testDataset = TrainingDataset[(int)(0.8f * TrainingDataset.Length)..TrainingDataset.Length];
            var testExpectedResultsDataset = ExpectedResultsDataset[(int)(0.8f * ExpectedResultsDataset.Length)..ExpectedResultsDataset.Length];

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
                    averageValidationLoss += GetError(data, testExpectedResultsDataset[l++]);
                }
                averageValidationLoss /= testExpectedResultsDataset.Length;

                Console.WriteLine($"Epoch {epoch++}: train loss: {averageTrainingLoss} | validation loss : {averageValidationLoss}");

                if (lastAverageValidationLoss < averageValidationLoss)
                {
                    Console.WriteLine($"End of training : overfitting threshold reached.");
                    return averageValidationLoss;
                }

                if (lastAverageValidationLoss == averageValidationLoss)
                {
                    Console.WriteLine($"End of training : learning ineffective. Increase number of neurons/layers or increase learning rate.");
                    return averageValidationLoss;
                }

                if (lastAverageValidationLoss - averageValidationLoss < MinLearningPrecision)
                {
                    Console.WriteLine($"End of training : learning inferior to minimal learning.");
                    return averageValidationLoss;
                }

                lastAverageValidationLoss = averageValidationLoss;
            }
        }

        #endregion

        #region Private Methods

        private double GetError(double[] inputs, double[] expectedOutputs)
        {
            InputLayer = new Layer(inputs);
            Layers[0].ComputeNeurons(InputLayer);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].ComputeNeurons(Layers[i - 1]);
            }
            return LossFunction(Layers.Last().GetOutputs(), expectedOutputs);
        }

        private double[][][] GetWeights()
        {
            double[][][] weights = new double[Layers.Length][][];
            for (int i = 0; i < Layers.Length; i++)
            {
                weights[i] = Layers[i].GetWeights();
            }
            return weights;
        }

        private void ModifyWeights(double[][][] weights)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ModifyWeights(weights[i]);
            }
        }

        private double TrainOnData(double[] inputs, double[] expectedOutputs)
        {
            double[][][] weights = GetWeights();
            for (int layerIndex = 0; layerIndex < Layers.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < Layers[layerIndex].Neurons.Length; neuronIndex++)
                {
                    for (int weightIndex = 0; weightIndex < Layers[layerIndex].Neurons[neuronIndex].Weights.Length; weightIndex++)
                    {
                        //Get error for positive change of the weight
                        weights[layerIndex][neuronIndex][weightIndex] += LearningRate;
                        ModifyWeights(weights);
                        double error = GetError(inputs, expectedOutputs);

                        //Get error for negative change of the weight
                        weights[layerIndex][neuronIndex][weightIndex] -= 2 * LearningRate;
                        ModifyWeights(weights);
                        double error2 = GetError(inputs, expectedOutputs);

                        //Calculate gradient of the error
                        var errorGradient = (error - error2) / (2 * LearningRate);

                        //Apply gradient descent to minimize error
                        weights[layerIndex][neuronIndex][weightIndex] += LearningRate - LearningRate * errorGradient;
                        ModifyWeights(weights);
                    }
                }
            }
            return GetError(inputs, expectedOutputs);
        }

        #endregion
    }
}
