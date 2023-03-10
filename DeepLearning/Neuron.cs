namespace DeepLearning
{
    public class Neuron
    {
        public double[] Weights { get; set; }

        public double Output { get; set; }

        private Func<double, double> ActivationFunction { get; set; }

        public Neuron(int numberOfInputs, Func<double, double> activationFunction)
        {
            Weights = new double[numberOfInputs + 1];
            for (int i = 0; i <= numberOfInputs; i++)
            {
                Weights[i] = new Random().NextDouble();
            }

            ActivationFunction = activationFunction;
        }

        public Neuron(double output)
        {
            Output = output;
        }

        public void ComputeOutput(double[] values)
        {
            double sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                sum += Weights[i] * values[i];
            }
            Output = ActivationFunction(sum);
        }
    }
}
