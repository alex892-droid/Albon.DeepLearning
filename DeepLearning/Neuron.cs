namespace Albon.DeepLearning
{
    internal class Neuron : INeuron
    {
        public double[] Weights { get; set; }

        public double Output { get; set; }

        private Func<double, double> ActivationFunction { get; set; }

        public Neuron(int numberOfInputs, Func<double, double> activationFunction)
        {
            Weights = new double[numberOfInputs];
            for (int weightIndex = 0; weightIndex < numberOfInputs; weightIndex++)
            {
                Weights[weightIndex] = MathTools.Random.NextDouble();
            }

            ActivationFunction = activationFunction;
        }

        public Neuron(double output)
        {
            Output = output;
        }

        public void ComputeOutput(double[] values)
        {
            if (values.Length != Weights.Length)
            {
                throw new ArgumentException("The number of input values does not match the number of weights.");
            }

            double sum = Weights.Zip(values, (weight, value) => weight * value).Sum();
            Output = ActivationFunction(sum);
        }
    }
}
