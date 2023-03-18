using Albon.DeepLearning.ActivationFunction;
using Albon.DeepLearning.Generator;
using Albon.DeepLearning.LossFunction;
using Albon.DeepLearning.Math;

namespace Albon.DeepLearning
{
    public class NeuralNetwork
    {
        private Layer InputLayer { get; set; }

        private Layer[] Layers { get; set; }

        private double[][] TrainingDataset { get; set; }

        private double[][] ExpectedResultsDataset { get; set; }

        private double MinLearningPrecision { get; set; }

        private ILossFunction LossFunction { get; set; }

        private IOptimizer Optimizer { get; set; }

        public NeuralNetwork(
            double[][] trainingDataset,
            double[][] expectedResultsDataset,
            int numberOfHiddenLayers,
            int numberOfNeuronsPerLayer,
            ILayerGenerator layerGenerator,
            ILossFunction lossFunction,
            IOptimizer optimizer,
            double minLearningPrecision = double.MinValue
            )
        {
            if (trainingDataset.Length < 100)
            {
                throw new ArgumentException("Not enough data to train a neural network.");
            }

            Layers = new Layer[numberOfHiddenLayers + 1];

            Layers = layerGenerator.GenerateLayers(trainingDataset[0].Length, numberOfNeuronsPerLayer, numberOfHiddenLayers, expectedResultsDataset[0].Length);

            TrainingDataset = trainingDataset;
            ExpectedResultsDataset = expectedResultsDataset;
            LossFunction = lossFunction;
            MinLearningPrecision = minLearningPrecision;
            Optimizer = optimizer;
        }

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
                    averageTrainingLoss += Optimizer.Optimize(data, trainingExpectedResultsDataset[k++], this);
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

        public double GetError(double[] inputs, double[] expectedOutputs)
        {
            InputLayer = new Layer(inputs);
            Layers[0].ComputeNeurons(InputLayer);
            for (int i = 1; i < Layers.Length; i++)
            {
                Layers[i].ComputeNeurons(Layers[i - 1]);
            }
            return LossFunction.EvaluateError(Layers.Last().GetOutputs(), expectedOutputs);
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
