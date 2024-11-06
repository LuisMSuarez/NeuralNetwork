using System.Diagnostics;

namespace NeuralNetworkLib
{
    public class Neuron
    {
        private int bias;
        private int? neuronValue;
        private readonly IEnumerable<Synapse> outgoingNeurons;
        private readonly IEnumerable<Synapse> incomingNeurons;
        private object syncRoot;

        public Neuron()
        {
            outgoingNeurons = new List<Synapse>();
            incomingNeurons = new List<Synapse>();
            bias = 0;
            this.neuronValue = null;
            this.syncRoot = new object();
        }

        /// <summary>
        /// Used to set a value when the neuron belongs to the input layer
        /// </summary>
        /// <param name="value"></param>
        public async Task SetValueAsync(int value)
        {
            this.neuronValue = value;
            await this.InvokeNextLayerAsync();
        }

        public void ConnectToNextLayer(Neuron successor, int weight)
        {
            var synapse = new Synapse(this, successor, weight);
            outgoingNeurons.Append(synapse);
            successor.incomingNeurons.Append(synapse);
        }

        private async void IncomingValueCallbackAsync(Synapse synapse)
        {
            var allIncomingValuesPresent = false;

            // Incoming callbacks can fire in parallel.  We need to have synchronization when testing if all incoming neurons have a value.
            lock (syncRoot)
            {
                // If we have recieved all inputs from all predecessors, we can calculate the value for this neuron.
                if (this.incomingNeurons.All(synapse => synapse.Value.HasValue))
                {
                    allIncomingValuesPresent = true;
                }
            }

            if (allIncomingValuesPresent)
            {
                this.neuronValue = this.incomingNeurons.Aggregate<Synapse, int>(
                    seed: 0,
                    (accumulatedValue, synapse) => accumulatedValue + synapse.Value!.Value)
                    + this.bias;
                Console.WriteLine($"Neuron has value {neuronValue}");
                await this.InvokeNextLayerAsync();
            }
        }

        public async Task InvokeNextLayerAsync()
        {
            Debug.Assert(this.neuronValue.HasValue);

            await Task.Run(() =>
            {
                Parallel.ForEach(outgoingNeurons, synapse =>
                    {
                        var synapseValue = this.neuronValue * synapse.Weight;
                        synapse.Value = synapseValue;
                        synapse.Destination!.IncomingValueCallbackAsync(synapse);
                    });
            });
        }
    }
}
