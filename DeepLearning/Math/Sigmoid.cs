namespace Albon.DeepLearning.Math
{
    public class Sigmoid : IActivationFunction
    {
        public double ActivationFunction(double input)
        {
            return 1.0 / (1.0 + System.Math.Exp(-input));
        }
    }
}
