import json

def translate_notebook(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    replacements = {
        "### 1. 自动微分（反向模式，动态图计算图）：": "### 1. Automatic Differentiation (Reverse Mode, Dynamic Computational Graph):",
        "通过 `Tensor` + 运算重载 + `backward()`，对模型输出 `F(x)` 求输入/参数的梯度。": "By using `Tensor` + Operator Overloading + `backward()`, compute gradients of inputs/parameters for model output `F(x)`.",
        "### 2. Integrated Gradients (IG) 归因：": "### 2. Integrated Gradients (IG) Attribution:",
        "对任意输入（标量/向量/矩阵）计算每个元素对标量输出的贡献度。": "Calculate the contribution of each element to the scalar output for any input (scalar/vector/matrix).",
        "- 若模型是用写的 `Tensor` 运算实现 → 用 **AD** 算梯度（快）": "- If the model is implemented using custom `Tensor` operations → Use **AD** for gradients (Fast)",
        "- 若模型是普通物理函数或黑盒函数 → 用 **FD 数值微分**算梯度（通用）": "- If the model is a standard physical function or black-box function → Use **FD (Finite Difference)** for gradients (General)",
        "### 3. 哪些运算支持自动微分（AD）？": "### 3. Which operations support Automatic Differentiation (AD)?",
        "在本实现中，**只有当某个运算满足以下两点**时，才支持自动微分（可参与计算图并可反传）：": "In this implementation, **only when an operation meets the following two points** is it supported by automatic differentiation (can participate in the computation graph and backpropagate):",
        "1) 该运算的前向会生成新的 `Tensor out`，并记录依赖关系 `out._prev`": "1) The forward pass of the operation generates a new `Tensor out` and records the dependency `out._prev`",
        "2) 为该运算实现了反向传播闭包 `out._backward`（定义如何把 `out.grad` 传回父节点）": "2) A backward propagation closure `out._backward` is implemented for the operation (defining how to pass `out.grad` back to parent nodes)",
        "**已支持的可微算子（会构图 + 可反传）**": "**Supported Differentiable Operators (Graph Construction + Backpropagation)**",
        "- 矩阵乘（MatMul）：`*(a::Tensor, b::Tensor)`": "- Matrix Multiplication (MatMul): `*(a::Tensor, b::Tensor)`",
        "- 加法（带广播反传）：`+(a::Tensor, b::Tensor)`": "- Addition (with broadcast backprop): `+(a::Tensor, b::Tensor)`",
        "- 减法（带广播反传）：`-(a::Tensor, b::Tensor)`": "- Subtraction (with broadcast backprop): `-(a::Tensor, b::Tensor)`",
        "- 逐元素乘（不重载 `.*`，改用函数）：`mul_elem(a::Tensor, b::Tensor)`": "- Element-wise Multiplication (does not overload `.*`, uses function): `mul_elem(a::Tensor, b::Tensor)`",
        "- 除以标量：`/(a::Tensor, b::Number)`（常用于 mean/平均 loss）": "- Division by Scalar: `/(a::Tensor, b::Number)` (Commonly used for mean/average loss)",
        "- ReLU 激活：`relu(t::Tensor)`": "- ReLU Activation: `relu(t::Tensor)`",
        "- 索引取标量（2D 单点）：`getindex(x::Tensor, i::Int, j::Int)`（用于选某个 logit 做 IG/求导）": "- Scalar Indexing (2D single point): `getindex(x::Tensor, i::Int, j::Int)` (Used to select a logit for IG/differentiation)",
        "- 全局求和到标量：`sum(x::Tensor)`": "- Global Sum to Scalar: `sum(x::Tensor)`",
        "### 4. 数值微分（FD，非自动微分）": "### 4. Numerical Differentiation (FD, Non-Automatic Differentiation)",
        "除自动微分（AD）外，本实现还提供 **数值微分（Finite Difference, FD）** 作为求梯度的备用方案，主要用于：": "In addition to Automatic Differentiation (AD), this implementation also provides **Numerical Differentiation (Finite Difference, FD)** as a backup solution for gradient calculation, mainly used for:",
        "- 模型是普通物理函数/黑盒函数（不使用 `Tensor` 构图）": "- The model is a standard physical function/black-box function (does not use `Tensor` for graph building)",
        "- 或模型包含尚未实现 `_backward` 的运算": "- Or the model contains operations that have not yet implemented `_backward`",
        "- 数值微分梯度函数：`grad_fd(model::Function, x::Array{Float64}; eps=1e-5)`": "- Numerical Differentiation Gradient Function: `grad_fd(model::Function, x::Array{Float64}; eps=1e-5)`",
        "- 在 IG 中通过 `method=:fd` 调用该函数，在每个插值点上用中心差分近似 \\(\\nabla_x F\\)（通用但计算成本较高）": "- Call this function via `method=:fd` in IG, approximating \\(\\nabla_x F\\) using central difference at each interpolation point (General but higher computational cost)",
        "### 4. 补充缺失算子 (Power, Inverse, Division, Scalar Mul)": "### 4. Supplement Missing Operators (Power, Inverse, Division, Scalar Mul)",
        "为了支持物理公式计算，我们需要补充定义的算子：`^` (power), `inv`, `1/x`, 以及标量乘法 `t * n`。": "To support physical formula calculations, we need to define additional operators: `^` (power), `inv`, `1/x`, and scalar multiplication `t * n`.",
        "验证我们的自动微分引擎在 $f(x) = \\sin(x) \\cdot x$ 上的精度。": "Verify the accuracy of our automatic differentiation engine on $f(x) = \\sin(x) \\cdot x$.",
        "比较 **|Custom_AD - Analytic|** 与 **|ForwardDiff - Analytic|** 的误差差距。": "Compare the error gap between **|Custom_AD - Analytic|** and **|ForwardDiff - Analytic|**."
    }
    
    for cell in nb['cells']:
        source = cell.get('source', [])
        new_source = []
        for line in source:
            processed_line = line
            for zh, en in replacements.items():
                if zh in processed_line:
                    processed_line = processed_line.replace(zh, en)
            new_source.append(processed_line)
        cell['source'] = new_source
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        
    print(f"Translated notebook saved to {output_path}")

if __name__ == "__main__":
    translate_notebook("ad_v7.ipynb", "ad_v7_en.ipynb")
