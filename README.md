Implementation of a Neural Network, with parallel execution neuron values as the partial results flow through the network.

The motivation for this project came from reading the [following article](https://towardsdatascience.com/understanding-llms-from-scratch-using-middle-school-math-e602d27ec876).  The article explains concepts of LLMs and Neural networks using an example that classifies specimens according to whether they represent leaves or flowers, based on attributes such as Color (as RGB vector) and volume.

The following image represents the neural network, and how it processes an input embedding to determine that the specimen is a 'leaf' and not a 'flower'.

![image](https://github.com/user-attachments/assets/3d2d56dd-5a9b-412c-8df1-f198a6d8f5c4)


As I read this article, I challenged myself to implement a model for the neural network, and an efficient implementation that would allow values to flow from one layer to the next as soon as they were computed.
The neural network is essentially an acyclic [directed graph](https://en.wikipedia.org/wiki/Directed_graph) where the vertices are the neurons and the edges represent connections between the neurons, that are labelled with weights.

I built the model using constructs such as:
* Neuron: Denotes a neuron in the neural network
* Synapse: Denotes an edge that connects 2 neurons (source and destination) with a weight
* Embedding: Vector of values that encode a particular datapoint to be run through the neural network
* Embedding matrix: Store of embeddings
  
As I approached this problem, I thought of ways to optimize calculation of values throughout the neural network, quickly realizing how it's possible to use parallelism. I used asynchronous programming in C# and synchronized access to critical resources using the [lock statement](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/statements/lock) to ensure I compute the value of a neuron only once, when the parallel threads of the neurons from the previous layer have finished computation of their value.
From the sample outputs below, we can see small variations in the order in which the values of the neurons were calculated in 3 separate executions, yet still producing the same output.

<img width="246" alt="image" src="https://github.com/user-attachments/assets/459848c3-5ad9-41a4-8ca5-085a9bc412f3">
<img width="255" alt="image" src="https://github.com/user-attachments/assets/46d11ecc-99d2-48a8-bfd9-f6153218e848">
<img width="255" alt="image" src="https://github.com/user-attachments/assets/519e934f-c26f-44e4-a3e3-43e246364638">

Constructs I used in this project:
* Async programming
* lock statements
* nullable types
* Properties
* LINQ extensions
* Callback pattern
