using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLib
{
    internal class Synapse
    {
        public int Weight { get; private set; }

        public Neuron Neuron { get; private set; }

        public int MyProperty { get; private set; }

        public Synapse(Neuron neuron, int weight) 
        {
            this.Neuron = neuron;
            this.Weight = weight;
        }
    }
}
