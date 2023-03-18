namespace Albon.DeepLearning.LossFunction
{
    public class BinaryCrossEntropyLoss : ILossFunction
    {
        public double EvaluateError(double[] predictedOutputs, double[] expectedOutputs)
        {
            double loss = 0.0;
            double epsilon = 1e-15;
            for (int i = 0; i < expectedOutputs.Length; i++)
            {
                double p = System.Math.Max(predictedOutputs[i], epsilon);
                double q = System.Math.Max(1 - predictedOutputs[i], epsilon);
                loss += expectedOutputs[i] * System.Math.Log(p) + (1 - expectedOutputs[i]) * System.Math.Log(q);
            }

            return -loss / expectedOutputs.Length;
        }
    }
}
