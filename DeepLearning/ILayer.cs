namespace Albon.DeepLearning
{
    internal interface ILayer
    {
        public void GenerateNeurons(int numberOfNeurons, int numberOfInputs);

        public void ComputeNeurons(Layer inputLayer);

        public double[] GetOutputs();

        public double[][] GetWeights();

        public void ModifyWeights(double[][] weights);
    }
}
