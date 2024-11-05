using System.Diagnostics;

namespace NeuralNetworkLib
{
    public class Neuron
    {
        private int bias;
        private int? neuronValue;
        private readonly IEnumerable<Synapse> outgoingNeurons;
        private readonly IEnumerable<Synapse> incomingNeurons;

        public Neuron()
        {
            outgoingNeurons = new List<Synapse>();
            incomingNeurons = new List<Synapse>();
            bias = 0;
            this.neuronValue = null;
        }

        /// <summary>
        /// Used to set a value when the neuron belongs to the input layer
        /// </summary>
        /// <param name="value"></param>
        public void SetValue(int value)
        {
            this.neuronValue = value;
        }

        public void ConnectToNextLayer(Neuron successor, int weight)
        {
            var synapse = new Synapse(this, successor, weight);
            outgoingNeurons.Append(synapse);
            successor.incomingNeurons.Append(synapse);
        }

        private void NotifyIncomingValue(Synapse synapse)
        {
            // If we have recieved all inputs from all predecessors, we can calculate the value for this neuron.
            if (this.incomingNeurons.All( synapse => synapse.Value.HasValue))
            {
                this.neuronValue = this.incomingNeurons.Aggregate<Synapse, int>(
                    seed: 0,
                    (accumulatedValue, synapse) => accumulatedValue + synapse.Value!.Value) 
                    + this.bias;
                this.InvokeNextLayer();
            }
        }

        public void InvokeNextLayer()
        {
            Debug.Assert(this.neuronValue.HasValue);

            foreach (var synapse in outgoingNeurons)
            {
                var synapseValue = this.neuronValue * synapse.Weight;
                synapse.Value = synapseValue;
                synapse.Destination!.NotifyIncomingValue(synapse);
            }
        }
    }
}
