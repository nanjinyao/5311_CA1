# MiniAD: A Pedagogical Automatic Differentiation Engine

MiniAD is a lightweight, reverse-mode automatic differentiation (AD) engine implemented from scratch in Julia. It is designed to demonstrate the core principles of AD and how it can be applied to interpret both neural networks and physics-based models using Integrated Gradients.

## Key Features
- **Define-by-Run:** Dynamic computational graph construction.
- **Custom Tensor Type:** Handles data and gradients automatically.
- **Supported Operations:** Matrix multiplication, ReLU, broadcasted arithmetic, and more.
- **Interpretability:** Built-in Integrated Gradients (IG) implementation.

## Project Structure
- `ad_v8.ipynb`: Main demonstration notebook.
- `MiniAD/`: Source code for the AD engine.
- `EE5311_Julia_CA1_Group6.pdf`: Project report.

## Requirements & Usage
- Julia 1.9+
- See `ad_v8.ipynb` for examples.
