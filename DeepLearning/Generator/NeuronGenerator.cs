using Albon.DeepLearning.ActivationFunction;

namespace Albon.DeepLearning.Generator
{
    public class NeuronGenerator : INeuronGenerator
    {
        public IActivationFunction ActivationFunction { get; set; }

        public NeuronGenerator(IActivationFunction activationFunction)
        {
            ActivationFunction = activationFunction;
        }

        Neuron INeuronGenerator.GenerateNeuron()
        {
            return new Neuron(ActivationFunction);
        }
    }
}
