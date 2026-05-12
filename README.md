# micrograd

Notes and code from studying Andrej Karpathy's [micrograd course](https://www.youtube.com/watch?v=VMj-3S1tku0).

This notebook builds a tiny autograd engine from scratch, starting with scalar derivatives and ending with a small neural network trained by gradient descent.

## What This Covers

- Derivatives, computational graphs, and the chain rule
- A scalar `Value` class that tracks data, gradients, parents, and local backward functions
- Manual backpropagation through simple expressions and a single neuron
- Topological ordering for automatic backpropagation
- A compact `Value.backward()` implementation
- A comparison with PyTorch autograd
- Simple `Neuron`, `Layer`, and `MLP` classes trained with squared-error loss

## Key Takeaways

- Backpropagation is the chain rule applied over a computational graph.
- Each operation only needs to know its local derivative.
- The forward pass builds the graph; the backward pass propagates gradients through it.
- Gradients must accumulate when one value affects the output through multiple paths.
- A neural network training loop is just forward pass, loss, backward pass, and parameter update.

## Run

Open `micrograd.ipynb` in Jupyter Notebook or VS Code and run the cells in order.

```bash
pip install numpy matplotlib graphviz torch
```

Graph rendering also requires the system Graphviz executable, including the `dot` command, to be available on PATH.
