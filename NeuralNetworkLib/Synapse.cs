namespace NeuralNetworkLib
{
    internal class Synapse
    {
        public int Weight { get; private set; }

        public int? Value { get;  set; }

        public Neuron Source { get;}

        public Neuron Destination { get;}

        public Synapse(Neuron source, Neuron destination, int weight)
        {
            this.Weight = weight;
            this.Value = null;
            this.Source = source;
            this.Destination = destination;
        }
    }
}
