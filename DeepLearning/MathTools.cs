﻿namespace DeepLearning
{
    public static class MathTools
    {
        public static readonly Random Random = new Random();

        #region Activation functions

        public static double Sigmoid(double x)
        {
            return 1.0 / (1.0 + System.Math.Exp(-x));
        }

        public static double Linear(double x)
        {
            return x;
        }

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }

        #endregion

        #region Loss/Cost functions

        public static double MeanSquaredError(double[] predictedOutput, double[] expectedOutput)
        {
            double mse = 0;
            for (int i = 0; i < predictedOutput.Length; i++)
            {
                mse += Math.Pow(expectedOutput[i] - predictedOutput[i], 2);
            }
            mse /= predictedOutput.Length;
            return mse;
        }

        public static double BinaryCrossEntropyLoss(double[] predictedOutput, double[] expectedOutput)
        {
            double loss = 0.0;

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                loss += expectedOutput[i] * Math.Log(predictedOutput[i]) - (1 - expectedOutput[i]) * Math.Log(1 - predictedOutput[i]);
            }

            return loss / expectedOutput.Length;
        }

        #endregion
    }
}
