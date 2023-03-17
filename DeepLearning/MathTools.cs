namespace Albon.DeepLearning
{
    public static class MathTools
    {
        public static readonly Random Random = new Random();

        #region Loss/Cost functions

        public static double BinaryCrossEntropyLoss(double[] predictedOutput, double[] expectedOutput)
        {
            double loss = 0.0;
            double epsilon = 1e-15;
            for (int i = 0; i < expectedOutput.Length; i++)
            {
                double p = System.Math.Max(predictedOutput[i], epsilon);
                double q = System.Math.Max(1 - predictedOutput[i], epsilon);
                loss += expectedOutput[i] * System.Math.Log(p) + (1 - expectedOutput[i]) * System.Math.Log(q);
            }

            return -loss / expectedOutput.Length;
        }

        #endregion
    }
}
