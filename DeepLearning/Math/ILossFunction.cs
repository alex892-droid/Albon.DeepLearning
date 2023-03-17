using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Math
{
    public interface ILossFunction
    {
        public double EvaluateError(double[] predictedOutputs, double[] expectedOutputs);
    }
}
