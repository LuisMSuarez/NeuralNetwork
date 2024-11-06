namespace NeuralNetworkLib
{
    using System.Numerics;

    public class Embedding
    {
        public required string Label { get; set; }
        public Vector<float> Embeddings { get; set; }
    }
}
