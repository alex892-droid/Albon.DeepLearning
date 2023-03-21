using Albon.DeepLearning.ActivationFunction;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Generator
{
    public interface ILayerGenerator
    {
        internal Layer[] GenerateLayers(int numberOfInputs, int numberOfOutputs);
    }
}
