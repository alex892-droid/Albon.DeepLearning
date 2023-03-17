namespace Albon.DeepLearning
{
    internal class Layer : ILayer
    {
        public Neuron[] Neurons { get; set; }

        public double Bias { get; set; }

        public Layer(int numberOfNeurons, int numberOfInputs, Func<double, double> activationFunction)
        {
            Neurons = new Neuron[numberOfNeurons];
            for (int i = 0; i < numberOfNeurons; i++)
            {
                Neurons[i] = new Neuron(numberOfInputs, activationFunction);
            }

            Bias = 1;
        }

        public Layer(double[] outputs)
        {
            Neurons = new Neuron[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                Neurons[i] = new Neuron(outputs[i]);
            }
        }

        public void ComputeNeurons(Layer inputLayer)
        {
            double[] inputValues = inputLayer.Neurons.Select(neuron => neuron.Output).Concat(new[] { inputLayer.Bias }).ToArray();

            foreach (Neuron neuron in Neurons)
            {
                neuron.ComputeOutput(inputValues);
            }
        }

        public double[] GetOutputs()
        {
            return Neurons.Select(neuron => neuron.Output).ToArray();
        }

        public double[][] GetWeights()
        {
            return Neurons.Select(neuron => neuron.Weights).ToArray();
        }

        public void ModifyWeights(double[][] weights)
        {
            if (weights.Length != Neurons.Length)
            {
                throw new ArgumentException("The number of weight arrays does not match the number of neurons in this layer.");
            }

            for (int i = 0; i < weights.Length; i++)
            {
                Neurons[i].Weights = weights[i];
            }
        }
    }
}
