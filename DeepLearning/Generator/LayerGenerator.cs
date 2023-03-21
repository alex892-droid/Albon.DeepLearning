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

        private int NumberOfNeuronsPerLayer { get; set; }

        private int NumberOfHiddenLayers { get; set; }

        public INeuronGenerator NeuronGeneratorLayer { get; set; }

        public INeuronGenerator NeuronGeneratorOutputs { get; set; }

        public LayerGenerator(INeuronGenerator neuronGeneratorLayer, INeuronGenerator neuronGeneratorOutputs, int numberOfNeuronsPerLayer, int numberOfHiddenLayers) 
        {
            NeuronGeneratorLayer = neuronGeneratorLayer;
            NeuronGeneratorOutputs = neuronGeneratorOutputs;
            NumberOfNeuronsPerLayer = numberOfNeuronsPerLayer;
            NumberOfHiddenLayers = numberOfHiddenLayers;
        }

        Layer[] ILayerGenerator.GenerateLayers(int numberOfInputs, int numberOfOutputs)
        {
            Layer[] layers = new Layer[NumberOfHiddenLayers + 1];

            //Initialization first hidden layer from number of inputs
            (layers[0] = new Layer(NeuronGeneratorLayer)).GenerateNeurons(NumberOfNeuronsPerLayer, numberOfInputs + NUMBER_OF_BIASES_PER_LAYER);

            for (int i = 1; i < NumberOfHiddenLayers; i++)
            {
                (layers[i] = new Layer(NeuronGeneratorLayer)).GenerateNeurons(NumberOfNeuronsPerLayer, NumberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER);
            }

            //Initialization output layer
            (layers[NumberOfHiddenLayers] = new Layer(NeuronGeneratorOutputs)).GenerateNeurons(numberOfOutputs, NumberOfNeuronsPerLayer + NUMBER_OF_BIASES_PER_LAYER);


            return layers;
        }
    }
}
