using System.Diagnostics;

namespace NeuralNetworkLib
{
    public class Neuron
    {
        private int bias;
        private double? neuronValue;
        private readonly List<Synapse> outgoingNeurons;
        private readonly List<Synapse> incomingNeurons;
        private object syncRoot;

        public Neuron()
        {
            outgoingNeurons = new List<Synapse>();
            incomingNeurons = new List<Synapse>();
            bias = 0;
            this.syncRoot = new object();
        }

        /// <summary>
        /// Used to set a value when the neuron belongs to the input layer
        /// </summary>
        /// <param name="value"></param>
        public async Task SetValueAsync(double value)
        {
            this.neuronValue = value;
            await this.InvokeNextLayerAsync();
        }

        public double? GetValue()
        {
            return this.neuronValue;
        }

        public void ConnectToNextLayer(Neuron successor, double weight)
        {
            var synapse = new Synapse(this, successor, weight);
            outgoingNeurons.Add(synapse);
            successor.incomingNeurons.Add(synapse);
        }

        private async Task IncomingValueCallbackAsync(Synapse synapse)
        {
            // Incoming callbacks can fire in parallel.  We need to have synchronization when testing if all incoming neurons have a value.
            bool invokeNextLayer = false;

            lock (syncRoot)
            {
                // If we have recieved all inputs from all predecessors, we can calculate the value for this neuron.
                if (this.incomingNeurons.All(synapse => synapse.Value.HasValue))
                {
                    invokeNextLayer = true;
                }
            }

            // Cannot call async inside a lock statement.
            if (invokeNextLayer)
            {
                // We treat the computation of the neuron value from all of the incoming synapses as a
                // "complex" operation that we want to run in the thread pool asynchronously.
                // If the PC has multiple cores, computation of this value can happen in parallel across cores.
                await Task.Run( async () =>
                {
                    this.neuronValue = this.incomingNeurons.Aggregate<Synapse, double>(
                            seed: 0,
                            (accumulatedValue, synapse) => accumulatedValue + synapse.Value!.Value)
                            + this.bias;
                    Console.WriteLine($"Neuron has value: {neuronValue}");

                    // Once this neuron has a computed value, we can invoke the next layer
                    await this.InvokeNextLayerAsync();
                });
            }
        }

        private async Task InvokeNextLayerAsync()
        {
            Debug.Assert(this.neuronValue.HasValue);
            Task[] nextLayerTasks = new Task[outgoingNeurons.Count];

            foreach(var index in Enumerable.Range(0, outgoingNeurons.Count))
            {
                // We treat the computation of the outgoing synapse values from this neuron's value and subsequent call of the next layer as a
                // "complex" operation that we want to run in the thread pool asynchronously.
                // If the PC has multiple cores, computation of can happen in parallel across cores.
                nextLayerTasks[index] = Task.Run(async () =>
                {
                    var synapse = outgoingNeurons[index];
                    var synapseValue = this.neuronValue * synapse.Weight;
                    synapse.Value = synapseValue;
                    await synapse.Destination!.IncomingValueCallbackAsync(synapse);
                });
            };

            await Task.WhenAll(nextLayerTasks);
        }
    }
}
