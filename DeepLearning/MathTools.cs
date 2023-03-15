namespace DeepLearning
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

        public static double BinaryCrossEntropyLoss(double[] y_true, double[] y_pred)
        {
            double epsilon = 1e-15; // a small constant to avoid taking the logarithm of zero or one
            double loss = 0.0;

            for (int i = 0; i < y_true.Length; i++)
            {
                loss += y_true[i] * Math.Log(y_pred[i] + epsilon) - (1 - y_true[i]) * Math.Log(1 - y_pred[i] + epsilon);
            }

            return loss / y_true.Length;
        }

        #endregion
    }
}
