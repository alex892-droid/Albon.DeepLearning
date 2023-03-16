using DeepLearning;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Math
{
    public class MeanSquaredError : ILossFunction
    {
        public Func<double[], double[], double> LossFunction => MathTools.MeanSquaredError;
    }
}
