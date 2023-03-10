using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearning
{
    public class NeuralNetwork
    {
        public const int NUMBER_OF_BIASES_PER_LAYER = 1;

        private Layer InputLayer { get; set; }

        private Layer[] Layers { get; set; }

        private int NumberOfNeuronsPerLayer { get; set; }

        public NeuralNetwork(int numberOfInputs, int numberOfOutputs, int numberOfHiddenLayers, int numberOfNeuronsPerLayer, Func<double, double> activationFunctionNeurons, Func<double, double> activationFunctionOutputs) 
        {
            Layers = new Layer[numberOfHiddenLayers + 1];

            //Initialization first hidden layer from number of inputs
            Layers[0] = new Layer(numberOfNeuronsPerLayer, numberOfInputs, activationFunctionNeurons);

            for (int i = 1; i < numberOfHiddenLayers; i++)
            {
                Layers[i] = new Layer(numberOfNeuronsPerLayer, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER, activationFunctionNeurons);
            }

            //Initialization output layer
            Layers[numberOfHiddenLayers] = new Layer(numberOfOutputs, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER, activationFunctionOutputs);

            NumberOfNeuronsPerLayer = numberOfHiddenLayers;
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
                Layers[i].ComputeNeurons(Layers[i-1]);
            }
            return Layers[^1];
        }

        public double Train(double[] inputs, double[] expectedOutputs, Func<double[], double[], double> costFunction, double learningRate)
        {
            double[][][] weights = GetWeights();
            for(int i = 0; i < Layers.Length; i++)
            {
                for(int j = 0; j < NumberOfNeuronsPerLayer; j++)
                {
                    int weightIndex = 0;
                    foreach(var weight in weights[i][j]) //Variable
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
