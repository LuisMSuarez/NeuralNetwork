using NeuralNetworkLib;
using System.Numerics;

namespace NeuralNetworkCmd
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var flowerEmbedding = new Embedding { Label = "Rose", Embeddings = new Vector<float>([241, 200, 4, 59.5f]) };
            var leafEmbedding = new Embedding { Label = "Maple", Embeddings = new Vector<float>([32, 107, 56, 11.2f]) };

            var embeddings = new EmbeddingMatrix();
            embeddings.AddEmbedding(flowerEmbedding);
            embeddings.AddEmbedding(leafEmbedding);
        }
    }
}