namespace Albon.DeepLearning.LossFunction
{
    public class MeanSquaredError : ILossFunction
    {
        public double EvaluateError(double[] predictedOutputs, double[] expectedOutputs)
        {
            double mse = 0;
            for (int i = 0; i < predictedOutputs.Length; i++)
            {
                mse += System.Math.Pow(expectedOutputs[i] - predictedOutputs[i], 2);
            }
            mse /= predictedOutputs.Length;
            return mse;
        }
    }
}
