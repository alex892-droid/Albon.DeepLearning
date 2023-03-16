using DeepLearning;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Math
{
    public class Linear : IActivationFunction
    {
        public Func<double, double> ActivationFunction => MathTools.Linear;
    }
}
