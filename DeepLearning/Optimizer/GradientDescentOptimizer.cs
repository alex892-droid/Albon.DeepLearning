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

        public double[][][] Optimize(double[][][] parameters, Func<double[][][], double> computeErrorMethod)
        {
            for (int layerIndex = 0; layerIndex < parameters.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < parameters[layerIndex].Length; neuronIndex++)
                {
                    for (int weightIndex = 0; weightIndex < parameters[layerIndex][neuronIndex].Length; weightIndex++)
                    {
                        //Get error for positive change of the weight
                        parameters[layerIndex][neuronIndex][weightIndex] += LearningRate;
                        double error = computeErrorMethod(parameters);

                        //Get error for negative change of the weight
                        parameters[layerIndex][neuronIndex][weightIndex] -= 2 * LearningRate;
                        double error2 = computeErrorMethod(parameters);

                        //Calculate gradient of the error
                        var errorGradient = (error - error2) / (2 * LearningRate);

                        //Apply gradient descent to minimize error
                        parameters[layerIndex][neuronIndex][weightIndex] += LearningRate - LearningRate * errorGradient;
                    }
                }
            }
            return parameters;
        }
    }
}
