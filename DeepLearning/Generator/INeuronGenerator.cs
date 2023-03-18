using Albon.DeepLearning.ActivationFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Generator
{
    public interface INeuronGenerator
    {
        internal Neuron GenerateNeuron();
    }
}
