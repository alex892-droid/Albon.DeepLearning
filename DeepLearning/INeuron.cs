namespace Albon.DeepLearning
{
    internal interface INeuron
    {
        public void InitializeWeights(int numberOfInputs);

        public void ComputeOutput(double[] values);
    }
}
