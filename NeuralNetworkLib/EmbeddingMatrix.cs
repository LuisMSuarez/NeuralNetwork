namespace NeuralNetworkLib
{
    public class EmbeddingMatrix
    {
        private List<Embedding> embeddings;

        public EmbeddingMatrix()
        {
            embeddings = new List<Embedding>();
        }

        public void AddEmbedding(Embedding embedding)
        {
            embeddings.Add(embedding);
        }

        public IEnumerable<Embedding> Embeddings { get { return embeddings.AsEnumerable(); } }
    }
}