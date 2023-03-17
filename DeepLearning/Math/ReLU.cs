namespace Albon.DeepLearning.Math
{
    public class ReLU : IActivationFunction
    {
        public double ActivationFunction(double input)
        {
            return System.Math.Max(0, input);
        }
    }
}
