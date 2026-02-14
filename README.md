# From Scratch: Reverse-Mode Automatic Differentiation and Integrated Gradients

![License](https://img.shields.io/badge/license-MIT-blue)
![Julia](https://img.shields.io/badge/julia-v1.9+-9558B2)
![Python](https://img.shields.io/badge/python-3.8+-3776AB)

This repository contains the source code for the project **"From Scratch: Reverse-Mode Automatic Differentiation and Integrated Gradients for Interpretability in Neural and Physics-Based Models"**, developed for the course **EE5311 Differentiable and Probabilistic Computing** at NUS.

## 📌 Project Overview

This project implements a lightweight **Reverse-Mode Automatic Differentiation (AD)** engine from first principles in Julia. It centers on a custom `Tensor` abstraction that supports:
- **Dynamic Computational Graphs**: Built via operator overloading.
- **Efficient Backpropagation**: Uses closure-based pullbacks and iterative DFS topological sorting (stack-safe for deep graphs).
- **Integrated Gradients (IG)**: A feature attribution method implemented from scratch to interpret both Neural Networks and Physical Models.

The project demonstrates the engine's capabilities through two main case studies:
1.  **Physics-Based Optimization**: Trajectory optimization for a robotic projectile (Netball).
2.  **Neural Network Interpretability**: Feature attribution for an MLP trained on the Iris dataset.

## 📂 File Structure

- **`ad_v7_en.ipynb`**: The core Jupyter Notebook containing the full implementation of the AD engine (`Tensor` struct, operators, `backward` engine), IG algorithm, and case studies (English version).
- **`generate_plots.jl`**: A pure Julia script version of the case studies, used to generate high-quality plots for the report.
- **`generate_plots.py`**: A Python reference implementation used to validate the Julia results.
- **`main.tex`**: The LaTeX source code for the project report.
- **`Julia_CA.pdf`**: The synthesized assignment PDF.
- **`word_count.py`, `extract_chinese.py`, `translate_notebook.py`**: Utility scripts for report statistics and translation.

## 🚀 Features

### 1. Custom AD Engine
- **Core Abstraction**: `Tensor` struct with `data`, `grad`, and `_backward` closure.
- **Operators**: Support for `+`, `-`, `*` (matmul), `/`, `sin`, `cos`, `exp`, `log`, `relu`, etc.
- **Graph Pruning**: Automatically prunes the graph for nodes with `requires_grad=false`.

### 2. Integrated Gradients (IG)
- Implements the path integral formulation:
  $$ IG_i(x) \approx (x_i - x_{0,i}) \times \frac{1}{m} \sum_{k=1}^{m} \nabla_x F\left(x_0 + \frac{k}{m}(x - x_0)\right) $$
- Supports both **AD mode** (for white-box `Tensor` models) and **Finite Difference mode** (for black-box functions).

## 🛠️ Requirements & Usage

### Julia Environment
To run the `.ipynb` or `.jl` files, you need Julia installed. The following packages are required:
```julia
using LinearAlgebra
using Statistics
using Plots
using Random
using IJulia # If running the notebook
```

### Python Environment (for validation scripts)
If you wish to run `generate_plots.py`:
```bash
pip install numpy matplotlib
```

## 👥 Contributors
- **Cao Yuan**
- **Liu Fei**
- **Jin Xuan**
- **Gao Jiaxuan**
- **Nan Jinyao**

## 📄 License
This project is open-source and available under the MIT License.
