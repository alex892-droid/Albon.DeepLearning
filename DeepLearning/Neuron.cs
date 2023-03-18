using Albon.DeepLearning.ActivationFunction;

namespace Albon.DeepLearning
{
    internal class Neuron : INeuron
    {
        public double[] Weights { get; set; }

        public double Output { get; set; }

        private IActivationFunction ActivationFunction { get; set; }

        public Neuron(IActivationFunction activationFunction)
        {
            ActivationFunction = activationFunction;
        }

        public Neuron(double output)
        {
            Output = output;
        }

        public void InitializeWeights(int numberOfInputs)
        {
            Weights = new double[numberOfInputs];
            for (int weightIndex = 0; weightIndex < numberOfInputs; weightIndex++)
            {
                Weights[weightIndex] = MathTools.Random.NextDouble();
            }
        }

        public void ComputeOutput(double[] values)
        {
            if (values.Length != Weights.Length)
            {
                throw new ArgumentException("The number of input values does not match the number of weights.");
            }

            double sum = Weights.Zip(values, (weight, value) => weight * value).Sum();
            Output = ActivationFunction.ActivationFunction(sum);
        }
    }
}
