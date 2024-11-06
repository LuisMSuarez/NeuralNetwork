using NeuralNetworkLib;
using System.Numerics;

namespace NeuralNetworkCmd
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            // This program uses a sample neural network described in this article:
            // https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876#:~:text=Llama%203.1.-,A%20simple%20neural%20network,-%3A

            var flowerEmbedding = new Embedding { Label = "Rose", Vector = new Vector<double>([241, 200, 4, 59.5]) };
            var leafEmbedding = new Embedding { Label = "Maple", Vector = new Vector<double>([32, 107, 56, 11.2f]) };

            var embeddings = new EmbeddingMatrix();
            embeddings.AddEmbedding(flowerEmbedding);
            embeddings.AddEmbedding(leafEmbedding);

            var embedding = embeddings.Embeddings.First();

            Neuron[] layer0neurons = [new Neuron(), new Neuron(), new Neuron(), new Neuron()];
            Neuron[] layer1neurons = [new Neuron(), new Neuron(), new Neuron()];
            Neuron[] layer2neurons = [new Neuron("Leaf"), new Neuron("Flower")];

            // Connect layer 0 neurons to layer 1 neurons
            layer0neurons[0].ConnectToNextLayer(layer1neurons[0], 0.10);
            layer0neurons[0].ConnectToNextLayer(layer1neurons[1], 0.12);
            layer0neurons[0].ConnectToNextLayer(layer1neurons[2], -0.36);

            layer0neurons[1].ConnectToNextLayer(layer1neurons[0], -0.29);
            layer0neurons[1].ConnectToNextLayer(layer1neurons[1], -0.05);
            layer0neurons[1].ConnectToNextLayer(layer1neurons[2], -0.21);

            layer0neurons[2].ConnectToNextLayer(layer1neurons[0], -0.07);
            layer0neurons[2].ConnectToNextLayer(layer1neurons[1], 0.04);
            layer0neurons[2].ConnectToNextLayer(layer1neurons[2], -0.27);

            layer0neurons[3].ConnectToNextLayer(layer1neurons[0], 0.46);
            layer0neurons[3].ConnectToNextLayer(layer1neurons[1], 0.16);
            layer0neurons[3].ConnectToNextLayer(layer1neurons[2], 0.18);

            // Connect layer 1 neurons to layer 2 neurons
            layer1neurons[0].ConnectToNextLayer(layer2neurons[0], -0.17);
            layer1neurons[0].ConnectToNextLayer(layer2neurons[1], 0.05);
            layer1neurons[1].ConnectToNextLayer(layer2neurons[0], 0.39);
            layer1neurons[1].ConnectToNextLayer(layer2neurons[1], -0.12);
            layer1neurons[2].ConnectToNextLayer(layer2neurons[0], 0.1);
            layer1neurons[2].ConnectToNextLayer(layer2neurons[1], -0.04);

            // Set input values for layer 0 from the embedding vector in parallel,
            // to kick off the neural network.
            var layer0Tasks = new List<Task>();
            foreach (var index in Enumerable.Range(0, layer0neurons.Length))
            {
                var neuron = layer0neurons[index];
                layer0Tasks.Add(neuron.SetValueAsync(embedding.Vector[index]));
            }

            // Calculation of the neural network is complete when propagation of values from the input layer all the way to the output layer completes.
            await Task.WhenAll(layer0Tasks);

            foreach (var neuron in layer2neurons)
            {
                Console.WriteLine($"Output layer neuron value {neuron.GetValue()!.Value}");
            }

            // The output neuron with the largest value determines the classification of the input.
            var classification = layer2neurons.MaxBy(neuron => neuron.GetValue());
            Console.WriteLine($"The input is a {classification!.Label}");
        }
    }
}