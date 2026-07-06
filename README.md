# Micrograd — Autograd Engine and Neural Network from Scratch

A scalar-valued automatic differentiation engine and neural network library built from scratch, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). Implements backpropagation, dynamic computation graphs, and gradient descent training without any ML frameworks.

## What This Implements

- **Automatic differentiation** — Each scalar value tracks its gradient through the computation graph, enabling backpropagation via reverse-mode autodiff
- **Dynamic computation graph** — Operations (`+`, `-`, `*`, `/`, `pow`, `tanh`) build a DAG on the fly; gradients are computed by topologically sorting and walking backward
- **Neural network primitives** — Neuron (weighted sum + bias + tanh activation), Layer, and Multi-Layer Perceptron (MLP), all composed from the autograd engine
- **Training loop** — Forward pass, MSE loss, backward pass, gradient descent parameter update

## Architecture

```
Value (scalar autograd)
  └── Neuron (weights · inputs + bias → tanh)
       └── Layer (collection of neurons)
            └── MLP (stack of layers)
```

Every arithmetic operation creates a new `Value` node linked to its inputs. Calling `FullBackward()` on the loss propagates gradients through the entire graph using the chain rule.

## Example — Binary Classification

```go
xs := [][]float64{
    {2.0, 3.0, -1.0},
    {3.0, -1.0, 0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0},
}
ys := []float64{1.0, -1.0, -1.0, 1.0}
mlp := engine.NewMLP([]int{4, 4, 1}, 3) // 3 inputs → 4 → 4 → 1
```

Training loop: forward pass → MSE loss → `loss.FullBackward()` → gradient descent → zero gradients → repeat.

## Project Structure

```
engine/
  value.go    — Core Value type: data, gradient, computation graph, backward pass
  neuron.go   — Single perceptron with weights, bias, tanh activation
  layer.go    — Collection of neurons
  mlp.go      — Multi-Layer Perceptron
main.go       — Demonstrations: autograd, single neuron, layer, full MLP training
```

## Run

```bash
go run main.go
```

## Acknowledgements

Inspired by [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) and his lectures on neural networks and backpropagation.
