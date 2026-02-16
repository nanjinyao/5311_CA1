module MiniAD

############################################################
# Mini AutoDiff + Chain + Integrated Gradients (IG)
############################################################

import Base: +, *, -, /, ^, getindex, size, length, show, sum, sin, cos, tan, exp, log
import Base.Broadcast: broadcasted, broadcastable, BroadcastStyle

export Tensor, backward, zero_grad!
export Linear, Chain, ReLU
export integrated_gradients, grad_fd, to_scalar
export register_broadcast_rule!

# ==========================================================
# 1) Core Tensor type
# ==========================================================

mutable struct Tensor
    data::Array{Float64}          # Numeric data (normalized to 2D)
    grad::Array{Float64}          # Gradient accumulator
    _backward::Function           # Pullback closure
    _prev::Set{Tensor}            # Parent nodes
    op::String                    # Operation name
    requires_grad::Bool           # Gradient tracking flag

    function Tensor(data; _children=(), _op="", requires_grad=true)
        # Normalize numeric inputs to 2D for consistency:
        # - scalar -> (1,1)
        # - vector -> (n,1)
        # - matrix -> unchanged
        d = if data isa Number
            reshape([Float64(data)], 1, 1)
        else
            arr = convert(Array{Float64}, data)
            ndims(arr) == 1 ? reshape(arr, :, 1) : arr
        end

        g = zeros(size(d))
        prev = requires_grad ? Set{Tensor}(_children) : Set{Tensor}()
        new(d, g, () -> nothing, prev, _op, requires_grad)
    end
end

Base.show(io::IO, t::Tensor) = print(io, "Tensor(shape=$(size(t.data)), op=\"$(t.op)\")")
Base.size(t::Tensor) = size(t.data)
Base.length(t::Tensor) = length(t.data)

zero_grad!(t::Tensor) = (t.grad .= 0.0)

# ==========================================================
# 2) Broadcast gradient reduction (unbroadcast)
# ==========================================================

function unbroadcast(out_grad::AbstractArray, orig_size::Tuple)
    g = out_grad

    # Reduce leading dimensions if broadcast created extra dims.
    while ndims(g) > length(orig_size)
        g = sum(g, dims=1)
    end

    # Reduce dimensions that were broadcast from size 1.
    for d in 1:length(orig_size)
        if orig_size[d] == 1 && size(g, d) != 1
            g = sum(g, dims=d)
        end
    end

    return reshape(g, orig_size)
end

# ==========================================================
# 3) Broadcast mechanism + extensible rule registry
# ==========================================================

struct TensorStyle <: BroadcastStyle end
Base.BroadcastStyle(::Type{Tensor}) = TensorStyle()
Base.BroadcastStyle(::TensorStyle, ::BroadcastStyle) = TensorStyle()
Base.BroadcastStyle(::BroadcastStyle, ::TensorStyle) = TensorStyle()

broadcastable(t::Tensor) = t

# Registry for broadcast backward rules:
# Key: function f (e.g., +, -, sin, exp)
# Val: rule(out::Tensor, res_data::Array{Float64}, args...) -> nothing
const BROADCAST_RULES = IdDict{Any,Function}()

"""
    register_broadcast_rule!(f, rule)

Register a backward rule for a broadcasted elementwise function `f`.

The `rule` signature must be:
    rule(out::Tensor, res_data, args...) -> nothing

- `out.grad` contains dL/d(out).
- `res_data` is forward result `f.(...)` (useful for exp/log, etc.).
- `args` are the original broadcast arguments (Tensor or Number/Array).
"""
function register_broadcast_rule!(f, rule::Function)
    BROADCAST_RULES[f] = rule
    return nothing
end

function broadcasted(::TensorStyle, f, args...)
    # Forward pass: unwrap Tensor -> data
    args_data = map(x -> x isa Tensor ? x.data : x, args)
    res_data = f.(args_data...)

    # Graph construction
    out_rg = any(x -> x isa Tensor && x.requires_grad, args)
    kids = filter(x -> x isa Tensor && x.requires_grad, args)
    out = Tensor(res_data, _children=kids, _op="Broadcast($(f))", requires_grad=out_rg)

    # Backward pass via registry
    if out_rg
        rule = get(BROADCAST_RULES, f, nothing)
        if rule === nothing
            # If gradients are requested but rule is missing, fail loudly.
            out._backward = () -> error("No broadcast backward rule registered for f = $(f)")
        else
            out._backward = () -> rule(out, res_data, args...)
        end
    end

    return out
end

# ==========================================================
# 4) Non-dot operator entry points (redirect to broadcasting)
# ==========================================================

Base.sin(x::Tensor) = sin.(x)
Base.cos(x::Tensor) = cos.(x)
Base.tan(x::Tensor) = tan.(x)
Base.exp(x::Tensor) = exp.(x)
Base.log(x::Tensor) = log.(x)


Base.:+(a::Tensor, b::Tensor) = a .+ b
Base.:+(a::Tensor, b::Number) = a .+ b
Base.:+(a::Number, b::Tensor) = a .+ b

Base.:-(a::Tensor, b::Tensor) = a .- b
Base.:-(a::Tensor, b::Number) = a .- b
Base.:-(a::Number, b::Tensor) = a .- b
Base.:-(a::Tensor) = 0.0 .- a

Base.:/(a::Tensor, b::Tensor) = a ./ b
Base.:/(a::Tensor, b::Number) = a ./ b
Base.:/(a::Number, b::Tensor) = a ./ b

# Note: *(Tensor,Tensor) is matrix multiplication (defined below).
# Tensor * Number is treated as elementwise scaling.
Base.:*(a::Tensor, b::Number) = a .* b
Base.:*(a::Number, b::Tensor) = a .* b

# Optional conveniences
Base.:^(a::Tensor, b::Number) = a .^ b
Base.:^(a::Tensor, b::Tensor) = a .^ b

# ==========================================================
# 5) Non-broadcast ops (MatMul / ReLU / Index / Sum)
# ==========================================================

function *(a::Tensor, b::Tensor)
    out_rg = a.requires_grad || b.requires_grad
    kids = filter(x -> x.requires_grad, (a, b))
    out = Tensor(a.data * b.data, _children=kids, _op="MatMul", requires_grad=out_rg)

    if out_rg
        function _backward()
            if a.requires_grad
                a.grad .+= out.grad * transpose(b.data)
            end
            if b.requires_grad
                b.grad .+= transpose(a.data) * out.grad
            end
        end
        out._backward = _backward
    end

    return out
end

function relu(t::Tensor)
    mask = t.data .> 0
    res = t.data .* mask

    out_rg = t.requires_grad
    out = Tensor(res, _children=(out_rg ? (t,) : ()), _op="ReLU", requires_grad=out_rg)

    if out_rg
        function _backward()
            t.grad .+= mask .* out.grad
        end
        out._backward = _backward
    end

    return out
end

function getindex(x::Tensor, i::Int, j::Int)
    out_rg = x.requires_grad
    out = Tensor(x.data[i, j], _children=(out_rg ? (x,) : ()), _op="GetIndex", requires_grad=out_rg)

    if out_rg
        function _backward()
            x.grad[i, j] += out.grad[1, 1]
        end
        out._backward = _backward
    end

    return out
end

function sum(x::Tensor)
    out_rg = x.requires_grad
    out = Tensor(Base.sum(x.data), _children=(out_rg ? (x,) : ()), _op="Sum", requires_grad=out_rg)

    if out_rg
        function _backward()
            x.grad .+= out.grad[1, 1] .* ones(size(x.data))
        end
        out._backward = _backward
    end

    return out
end

# ==========================================================
# 6) Backprop engine (iterative topo replay)
# ==========================================================

function backward_iterative(root::Tensor; init_grad=nothing)
    if !root.requires_grad
        return nothing
    end

    topo = Tensor[]
    visited = Set{Tensor}()
    stack = [(root, false)]

    while !isempty(stack)
        curr, processed = pop!(stack)
        if curr in visited
            continue
        end
        if processed
            push!(visited, curr)
            push!(topo, curr)
        else
            push!(stack, (curr, true))
            for parent in curr._prev
                if parent ∉ visited
                    push!(stack, (parent, false))
                end
            end
        end
    end

    root.grad .= (init_grad === nothing ? 1.0 : init_grad)

    for node in reverse(topo)
        node._backward()
    end

    return nothing
end

const backward = backward_iterative

# ==========================================================
# 7) Layers + Chain
# ==========================================================

struct Linear
    W::Tensor
    b::Tensor
end

function Linear(inf::Int, outf::Int; init=:xavier)
    scale = init == :xavier ? sqrt(2.0 / (inf + outf)) : 0.01
    W = Tensor(randn(outf, inf) .* scale, _op="W")
    b = Tensor(zeros(outf, 1), _op="b")
    return Linear(W, b)
end

(layer::Linear)(x::Tensor) = layer.W * x .+ layer.b

struct ReLU end
(layer::ReLU)(x::Tensor) = relu(x)

struct Chain
    layers::Vector{Any}
end
Chain(layers...) = Chain(Any[layers...])

function (m::Chain)(x::Tensor)
    for layer in m.layers
        x = layer(x)
    end
    return x
end

zero_grad!(m::Linear) = (m.W.grad .= 0.0; m.b.grad .= 0.0)

function zero_grad!(m::Chain)
    for layer in m.layers
        if layer isa Linear
            zero_grad!(layer)
        end
    end
end

# ==========================================================
# 8) Helpers & Integrated Gradients
# ==========================================================

function to_scalar(out::Tensor; target::Union{Nothing,Int}=nothing)
    if size(out.data) == (1, 1)
        return out
    end
    return target === nothing ? sum(out) : out[target, 1]
end

function grad_fd(model::Function, x::Array{Float64}; eps::Float64=1e-5)
    g = zeros(size(x))
    fx = model(x)
    for idx in eachindex(x)
        old = x[idx]
        x[idx] = old + eps
        f1 = model(x)
        x[idx] = old - eps
        f2 = model(x)
        x[idx] = old
        g[idx] = (f1 - f2) / (2eps)
    end
    return g, fx
end

# Shape alignment helper:
# - scalar -> (1,1)
# - vector -> (n,1)
# - matrix -> (n,m)
function _align2d(a)
    if a isa Number
        return reshape([Float64(a)], 1, 1)
    else
        arr = convert(Array{Float64}, a)
        return ndims(arr) == 1 ? reshape(arr, :, 1) : arr
    end
end

function integrated_gradients(model, input, baseline; steps::Int=50, target=nothing, method=:ad)
    x = _align2d(input)
    x0 = _align2d(baseline)

    diff = x .- x0
    total_grads = zeros(size(x))

    for s in 1:steps
        α = s / steps
        z = x0 .+ α .* diff

        if method == :ad
            xt = Tensor(z, _op="Input")
            out = model(xt)
            y = to_scalar(out; target=target)

            if model isa Chain
                zero_grad!(model)
            end
            xt.grad .= 0.0
            y.grad .= 0.0

            backward(y)
            total_grads .+= xt.grad

        elseif method == :fd
            g, _ = grad_fd(model, copy(z))
            total_grads .+= g
        end
    end

    avg_grads = total_grads ./ steps
    return diff .* avg_grads
end

# ==========================================================
# 9) Default broadcast rules registration
# ==========================================================
# Each rule receives:
#   out      : Tensor (contains out.grad)
#   res_data : forward result array
#   args...  : original broadcast args (Tensor or Number/Array)

function init_broadcast_rules!()
    # Clear existing rules (safe for re-init during development)
    empty!(BROADCAST_RULES)

    register_broadcast_rule!(+, (out, res, args...) -> begin
        for x in args
            if x isa Tensor && x.requires_grad
                x.grad .+= unbroadcast(out.grad, size(x.data))
            end
        end
        nothing
    end)

    register_broadcast_rule!(-, (out, res, args...) -> begin
        x, y = args
        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad, size(x.data))
        end
        if y isa Tensor && y.requires_grad
            y.grad .-= unbroadcast(out.grad, size(y.data))
        end
        nothing
    end)

    register_broadcast_rule!(*, (out, res, args...) -> begin
        x, y = args
        if x isa Tensor && x.requires_grad
            val_y = y isa Tensor ? y.data : y
            x.grad .+= unbroadcast(out.grad .* val_y, size(x.data))
        end
        if y isa Tensor && y.requires_grad
            val_x = x isa Tensor ? x.data : x
            y.grad .+= unbroadcast(out.grad .* val_x, size(y.data))
        end
        nothing
    end)

    register_broadcast_rule!(/, (out, res, args...) -> begin
        x, y = args
        val_x = x isa Tensor ? x.data : x
        val_y = y isa Tensor ? y.data : y

        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad ./ val_y, size(x.data))
        end
        if y isa Tensor && y.requires_grad
            y.grad .+= unbroadcast(out.grad .* (-val_x ./ (val_y .^ 2)), size(y.data))
        end
        nothing
    end)

    register_broadcast_rule!(^, (out, res, args...) -> begin
        x, y = args
        val_x = x isa Tensor ? x.data : x
        val_y = y isa Tensor ? y.data : y

        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad .* (val_y .* (val_x .^ (val_y .- 1))), size(x.data))
        end
        nothing
    end)

    register_broadcast_rule!(sin, (out, res, args...) -> begin
        x = args[1]
        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad .* cos.(x.data), size(x.data))
        end
        nothing
    end)

    register_broadcast_rule!(cos, (out, res, args...) -> begin
        x = args[1]
        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad .* (-sin.(x.data)), size(x.data))
        end
        nothing
    end)

    register_broadcast_rule!(tan, (out, res, args...) -> begin
        x = args[1]
        if x isa Tensor && x.requires_grad
            # d/dx tan(x) = 1 / cos(x)^2
            x.grad .+= unbroadcast(out.grad ./ (cos.(x.data) .^ 2), size(x.data))
        end
        nothing
    end)

    register_broadcast_rule!(exp, (out, res, args...) -> begin
        x = args[1]
        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad .* res, size(x.data))
        end
        nothing
    end)

    register_broadcast_rule!(log, (out, res, args...) -> begin
        x = args[1]
        if x isa Tensor && x.requires_grad
            x.grad .+= unbroadcast(out.grad ./ x.data, size(x.data))
        end
        nothing
    end)

    return nothing
end

# Ensure rules are registered when the package is loaded.
function __init__()
    init_broadcast_rules!()
end

end # module
