namespace NeuralNetworkLib
{
    internal class Synapse
    {
        public double Weight { get; private set; }

        public double? Value { get;  set; }

        public Neuron Source { get;}

        public Neuron Destination { get;}

        public Synapse(Neuron source, Neuron destination, double weight)
        {
            this.Weight = weight;
            this.Value = null;
            this.Source = source;
            this.Destination = destination;
        }
    }
}
