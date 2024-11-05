namespace NeuralNetworkLib
{
    public class Neuron
    {
        private int bias;
        private int value;
        private readonly IEnumerable<Synapse> successors;
        private readonly IEnumerable<Neuron> predecessors;
        private readonly IEnumerable<int> values;

        public Neuron()
        {
            successors = new List<Synapse>();
            predecessors = new List<Neuron>();
            this.values = new List<int>();
            bias = 0;
        }

        /// <summary>
        /// Used to set a value when the neuron belongs to the input layer
        /// </summary>
        /// <param name="value"></param>
        public void SetValue(int value)
        {
            this.value = value;
        }

        public void ConnectToSuccessor(Neuron successor, int weight)
        {
            var synapse = new Synapse(successor, weight);
            successors.Append(synapse);
            successor.ConnectToPredecessor(this);
        }

        public void ConnectToPredecessor(Neuron predecessor)
        {
            predecessors.Append(predecessor);
        }

        private void InputValue(int value)
        {
            this.values.Append(value);

            // If we have recieved all inputs from all predecessors, we can calculate the value for this neuron
            if (this.values.Count() == this.predecessors.Count())
            {
                this.value = this.values.Aggregate((item, accumulated) => accumulated + item) + this.bias;
                this.InvokeNextLayer();
            }
        }

        public void InvokeNextLayer()
        {
            foreach (var synapse in successors)
            {
                var synapseValue = this.value * synapse.Weight;
                synapse.Neuron.InputValue(synapseValue);
            }
        }
    }
}
