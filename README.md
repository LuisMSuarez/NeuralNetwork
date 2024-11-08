Implementation of a Neural Network, with parallel execution neuron values as the partial results flow through the network.

The motivation for this project came from reading the [following article](https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876).  The article explains concepts of LLMs and Neural netoworks using an example that classifies inputs according to whether they represent leaves or flowers, based on attributes such as Color (as RGB vector) and volume.

As I read this article, I challenged myself to implement a model for the neural network, and an efficient implementation that would allow values to flow from one layer to the next as soon as they became available.
A neural network is essentially an acyclic [directed graph](https://en.wikipedia.org/wiki/Directed_graph) where the vertices are the neurons and the edges represent connections between the neurons, that are labelled with weights.

I built the model using constructs such as:
* Neuron: Denotes a neuron in the neural network
* Synapse: Denotes an edge that connects 2 neurons (source and destination) with a weight
* Embedding: Vector of values that encode a particular datapoint to be run through the neural network
* Embedding matrix: Store of embeddings
  
As I approached this problem, I thought of ways to optimize calculation of values throughout the neural network, quickly realizing how it's possible to use parallelism. I used asynchronous programming in C# and synchronized access to critical resources using the [lock statement](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/statements/lock) to ensure I compute the value of a neuron only once the parallel threads of the neurons from the previous layer (that connect to a given neuron) have computed a value.

Constructs I used in this project:
* Async programming
* lock statements
* nullable types
* Properties
* LINQ
