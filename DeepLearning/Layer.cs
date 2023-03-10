namespace DeepLearning
{
    public class Layer
    {
        private Neuron[] Neurons { get; set; }

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
            double[] inputValues = new double[inputLayer.Neurons.Length + 1];
            for (int i = 0; i < inputLayer.Neurons.Length; i++)
            {
                inputValues[i] = inputLayer.Neurons[i].Output;
            }
            inputValues[inputLayer.Neurons.Length] = inputLayer.Bias;

            foreach (Neuron neuron in Neurons)
            {
                neuron.ComputeOutput(inputValues);
            }
        }

        public double[] GetOutputs()
        {
            double[] values = new double[Neurons.Length];
            for (int i = 0; i < Neurons.Length; i++)
            {
                values[i] = Neurons[i].Output;
            }
            return values;
        }

        public double[][] GetWeights()
        {
            double[][] values = new double[Neurons.Length][];
            for (int i = 0; i < Neurons.Length; i++)
            {
                values[i] = Neurons[i].Weights;
            }
            return values;
        }

        public void ModifyWeights(double[][] weights)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                Neurons[i].Weights = weights[i];
            }
        }
    }
}
