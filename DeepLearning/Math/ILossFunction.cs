using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Albon.DeepLearning.Math
{
    public interface ILossFunction
    {
        public Func<double[], double[], double> LossFunction { get; }
    }
}
