using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Math
{
    public class GradientDescentOptimizer : IOptimizer
    {
        public double LearningRate { get; set; }

        public GradientDescentOptimizer(double learningRate)
        { 
            LearningRate = learningRate;
        }

        public double Optimize(double[] inputs, double[] expectedOutputs, NeuralNetwork neuralNetwork)
        {
            double[][][] weights = neuralNetwork.GetWeights();
            for (int layerIndex = 0; layerIndex < weights.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < weights[layerIndex].Length; neuronIndex++)
                {
                    for (int weightIndex = 0; weightIndex < weights[layerIndex][neuronIndex].Length; weightIndex++)
                    {
                        //Get error for positive change of the weight
                        weights[layerIndex][neuronIndex][weightIndex] += LearningRate;
                        neuralNetwork.ModifyWeights(weights);
                        double error = neuralNetwork.GetError(inputs, expectedOutputs);

                        //Get error for negative change of the weight
                        weights[layerIndex][neuronIndex][weightIndex] -= 2 * LearningRate;
                        neuralNetwork.ModifyWeights(weights);
                        double error2 = neuralNetwork.GetError(inputs, expectedOutputs);

                        //Calculate gradient of the error
                        var errorGradient = (error - error2) / (2 * LearningRate);

                        //Apply gradient descent to minimize error
                        weights[layerIndex][neuronIndex][weightIndex] += LearningRate - LearningRate * errorGradient;
                        neuralNetwork.ModifyWeights(weights);
                    }
                }
            }
            return neuralNetwork.GetError(inputs, expectedOutputs);
        }
    }
}
