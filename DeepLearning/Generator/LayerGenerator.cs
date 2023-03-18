using Albon.DeepLearning.ActivationFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Generator
{
    public class LayerGenerator : ILayerGenerator
    {
        private const int NUMBER_OF_BIASES_PER_LAYER = 1;

        public INeuronGenerator NeuronGeneratorLayer { get; set; }

        public INeuronGenerator NeuronGeneratorOutputs { get; set; }

        public LayerGenerator(INeuronGenerator neuronGeneratorLayer, INeuronGenerator neuronGeneratorOutputs) 
        {
            NeuronGeneratorLayer = neuronGeneratorLayer;
            NeuronGeneratorOutputs = neuronGeneratorOutputs;
        }

        Layer[] ILayerGenerator.GenerateLayers(int numberOfInputs, int numberOfNeuronsPerLayer, int numberOfHiddenLayers, int numberOfResults)
        {
            Layer[] layers = new Layer[numberOfHiddenLayers + 1];

            //Initialization first hidden layer from number of inputs
            (layers[0] = new Layer(NeuronGeneratorLayer)).GenerateNeurons(numberOfNeuronsPerLayer, numberOfInputs + NUMBER_OF_BIASES_PER_LAYER);

            for (int i = 1; i < numberOfHiddenLayers; i++)
            {
                (layers[i] = new Layer(NeuronGeneratorLayer)).GenerateNeurons(numberOfNeuronsPerLayer, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER);
            }

            //Initialization output layer
            (layers[numberOfHiddenLayers] = new Layer(NeuronGeneratorOutputs)).GenerateNeurons(numberOfResults, numberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER);


            return layers;
        }
    }
}
