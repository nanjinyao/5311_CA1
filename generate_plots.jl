using LinearAlgebra
using Statistics
using Plots
using Random

ENV["GKSwstype"] = "100" # Headless mode

# Set seed for reproducibility
Random.seed!(42)

println("--- initializing AD engine ---")

# =========================================================================
# 1. CORE AD ENGINE
# =========================================================================

mutable struct Tensor
    data::Array{Float64}
    grad::Array{Float64}
    _backward::Function
    _prev::Set{Tensor}
    op::String
    requires_grad::Bool

    function Tensor(data; _children=(), _op="", requires_grad=true)
        d = data isa Number ? reshape([Float64(data)], 1, 1) : convert(Array{Float64}, data)
        # Ensure 2D column vector for 1D arrays
        if ndims(d) == 1
             d = reshape(d, :, 1)
        end
        g = zeros(size(d))
        prev = requires_grad ? Set{Tensor}(_children) : Set{Tensor}()
        new(d, g, () -> nothing, prev, _op, requires_grad)
    end
end

# Helper to creates a new tensor from raw data
function tensor(data; requires_grad=true)
    return Tensor(data; requires_grad=requires_grad)
end

import Base: +, -, *, /, ^, sin, cos, exp, log, sum, max, getindex

# --- Operations ---

# Broadcast helper
function unbroadcast(out_grad::Array{Float64}, orig_size::Tuple)
    g = out_grad
    os = orig_size
    while length(os) < ndims(g)
        os = (1, os...)
    end
    for d in 1:ndims(g)
        if os[d] == 1 && size(g, d) > 1
            g = Base.sum(g, dims=d)
        end
    end
    return reshape(g, os)
end

function +(a::Tensor, b::Tensor)
    out_rg = a.requires_grad || b.requires_grad
    kids = filter(x -> x.requires_grad, (a, b))
    out = Tensor(a.data .+ b.data, _children=kids, _op="+", requires_grad=out_rg)
    if out_rg
        function _backward()
            if a.requires_grad a.grad .+= unbroadcast(out.grad, size(a.data)) end
            if b.requires_grad b.grad .+= unbroadcast(out.grad, size(b.data)) end
        end
        out._backward = _backward
    end
    return out
end

function *(a::Tensor, b::Tensor)
    # Matmul
    out_rg = a.requires_grad || b.requires_grad
    kids = filter(x -> x.requires_grad, (a, b))
    out = Tensor(a.data * b.data, _children=kids, _op="MatMul", requires_grad=out_rg)
    if out_rg
        function _backward()
            if a.requires_grad a.grad .+= out.grad * transpose(b.data) end
            if b.requires_grad b.grad .+= transpose(a.data) * out.grad end
        end
        out._backward = _backward
    end
    return out
end

function -(a::Tensor, b::Tensor)
    out_rg = a.requires_grad || b.requires_grad
    kids = filter(x -> x.requires_grad, (a, b))
    out = Tensor(a.data .- b.data, _children=kids, _op="-", requires_grad=out_rg)
    if out_rg
        function _backward()
            if a.requires_grad a.grad .+= unbroadcast(out.grad, size(a.data)) end
            if b.requires_grad b.grad .-= unbroadcast(out.grad, size(b.data)) end
        end
        out._backward = _backward
    end
    return out
end

function ^(a::Tensor, k::Number)
    out_rg = a.requires_grad
    out = Tensor(a.data .^ k, _children=(out_rg ? (a,) : ()), _op="^$k", requires_grad=out_rg)
    if out_rg
        function _backward()
            a.grad .+= (k .* a.data .^ (k - 1)) .* out.grad
        end
        out._backward = _backward
    end
    return out
end

function sin(a::Tensor)
    out_rg = a.requires_grad
    out = Tensor(sin.(a.data), _children=(out_rg ? (a,) : ()), _op="sin", requires_grad=out_rg)
    if out_rg
        function _backward()
            a.grad .+= cos.(a.data) .* out.grad
        end
        out._backward = _backward
    end
    return out
end

function cos(a::Tensor)
    out_rg = a.requires_grad
    out = Tensor(cos.(a.data), _children=(out_rg ? (a,) : ()), _op="cos", requires_grad=out_rg)
    if out_rg
        function _backward()
            a.grad .+= (-sin.(a.data)) .* out.grad
        end
        out._backward = _backward
    end
    return out
end

function relu(a::Tensor)
    out_rg = a.requires_grad
    out_data = max.(0.0, a.data)
    out = Tensor(out_data, _children=(out_rg ? (a,) : ()), _op="ReLU", requires_grad=out_rg)
    if out_rg
        function _backward()
            a.grad .+= (a.data .> 0) .* out.grad
        end
        out._backward = _backward
    end
    return out
end

function exp(a::Tensor)
    out_rg = a.requires_grad
    out = Tensor(exp.(a.data), _children=(out_rg ? (a,) : ()), _op="exp", requires_grad=out_rg)
    if out_rg
        function _backward()
            a.grad .+= out.data .* out.grad
        end
        out._backward = _backward
    end
    return out
end

function log(a::Tensor)
    out_rg = a.requires_grad
    out = Tensor(log.(a.data), _children=(out_rg ? (a,) : ()), _op="log", requires_grad=out_rg)
    if out_rg
        function _backward()
            a.grad .+= (1.0 ./ a.data) .* out.grad
        end
        out._backward = _backward
    end
    return out
end

# Scalar support
+(a::Tensor, b::Number) = a + Tensor(b, requires_grad=false)
+(a::Number, b::Tensor) = Tensor(a, requires_grad=false) + b
-(a::Tensor, b::Number) = a - Tensor(b, requires_grad=false)
-(a::Number, b::Tensor) = Tensor(a, requires_grad=false) - b
*(a::Tensor, b::Number) = Tensor(a.data .* b, _children=(a.requires_grad ? (a,) : ()), _op="*s", requires_grad=a.requires_grad)
function *(n::Number, t::Tensor)
    out_rg = t.requires_grad
    out = Tensor(n .* t.data, _children=(out_rg ? (t,) : ()), _op="s*", requires_grad=out_rg)
    if out_rg
        function _backward() t.grad .+= n .* out.grad end
        out._backward = _backward
    end
    return out
end

# Indexing support (Critical for IG wrappers)
function getindex(t::Tensor, i::Int)
    out_rg = t.requires_grad
    out = Tensor(t.data[i:i, :], _children=(out_rg ? (t,) : ()), _op="idx", requires_grad=out_rg)
    if out_rg
        function _backward()
            t.grad[i] += out.grad[1]
        end
        out._backward = _backward
    end
    return out
end

# --- Backward Pass ---

function backward_iterative(root::Tensor)
    if !root.requires_grad return nothing end
    topo = Tensor[]
    visited = Set{Tensor}()
    stack = [(root, false)]
    
    while !isempty(stack)
        curr, processed = pop!(stack)
        if curr in visited continue end
        
        if processed
            push!(visited, curr)
            push!(topo, curr)
        else
            push!(stack, (curr, true))
            for parent in curr._prev
                if parent ∉ visited push!(stack, (parent, false)) end
            end
        end
    end
    
    root.grad .= 1.0
    for node in reverse(topo)
        node._backward()
    end
end

function zero_grad!(t::Tensor)
    t.grad .= 0.0
    nothing
end

# =========================================================================
# 2. INTEGRATED GRADIENTS
# =========================================================================

function integrated_gradients(model, input, baseline; steps=50, target=nothing)
    x = input isa Number ? reshape([Float64(input)], 1, 1) : convert(Array{Float64}, input)
    x0 = baseline isa Number ? reshape([Float64(baseline)], 1, 1) : convert(Array{Float64}, baseline)
    
    diff = x .- x0
    total_grads = zeros(size(x))
    
    for s in 1:steps
        alpha = s / steps
        z_data = x0 .+ alpha .* diff
        z = Tensor(z_data, _op="Input")
        
        out = model(z)
        
        backward_iterative(out)
        total_grads .+= z.grad
    end
    
    avg_grads = total_grads ./ steps
    attrs = diff .* avg_grads
    return attrs
end

# =========================================================================
# 3. PLOTTING SETUP
# =========================================================================

default(
    fontfamily="Computer Modern",
    linewidth=2,
    framestyle=:box,
    label=nothing,
    grid=:true,
    dpi=300,
    size=(700, 500), # Increased size
    margin=5Plots.mm
)

# =========================================================================
# 4. CASE STUDY 1: NETBALL OPTIMIZATION
# =========================================================================
println("\n--- Running Netball Optimization ---")
println("Starting optimization loop...")
v_hist = Float64[]
theta_hist = Float64[]
try
    for i in 1:3000
        # Re-build graph per iteration
        v_t = Tensor(v_param.data, _op="v")
        th_t = Tensor(theta_param.data, _op="th")
        
        theta_rad = th_t * (3.1415926535 / 180.0)
        
        # Breakdown for debugging if needed
        sin_th = sin(theta_rad)
        cos_th = cos(theta_rad)
        
        # term_rise
        # (d * sin) * (cos^-1)
        term_rise = (d_dist * sin_th) * (cos_th ^ -1)
        
        # term_drop
        # num * (denom^-1)
        # denom = v^2 * cos^2
        denom = (v_t ^ 2) * (cos_th ^ 2)
        num = 0.5 * g_grav * (d_dist ^ 2)
        
        # scalar * tensor
        term_drop = num * (denom ^ -1)
        
        # Sum
        # Tensor([y]) + T + T
        h_pred = Tensor([y0_bot]) + term_rise + term_drop
        
        loss = (h_pred - h_hoop) ^ 2
        
        if i % 100 == 0
            push!(v_hist, v_t.data[1])
            push!(theta_hist, th_t.data[1])
            # println("Iter $i: Loss=$(loss.data[1]) v=$(v_t.data[1]) th=$(th_t.data[1])")
        end
        
        backward_iterative(loss)
        
        # SGD
        # Check gradients
        if any(isnan, v_t.grad) || any(isnan, th_t.grad)
            println("NaN in grads at iter $i")
            break
        end
        
        v_param.data .-= lr .* v_t.grad
        theta_param.data .-= lr .* th_t.grad
    end
catch e
    println("Error in optimization loop:")
    showerror(stdout, e)
    println()
    # for (exc, bt) in Base.catch_stack()
    #    showerror(stdout, exc, bt)
    #    println()
    # end
end


# Define defaults in case of failure
if !@isdefined(v_opt)
    v_opt = 10.0
end
if !@isdefined(theta_opt)
    theta_opt = 0.0
end

println("Optimized: v=$(round(v_opt, digits=3)), theta=$(round(theta_opt, digits=3))")

# Plot
function get_traj(v, deg)
    rad = deg * π / 180.0
    # Guard against NaN/Inf
    if isnan(v) v = 10.0 end
    if isnan(deg) deg = 0.0 end
    
    T = d_dist / (v * cos(rad))
    if T < 0 || isnan(T) || T > 100
        T = 1.0
    end
    ts = 0:0.01:T
    xs = [v * cos(rad) * t for t in ts]
    ys = [y0_bot + v * sin(rad) * t + 0.5 * g_grav * t^2 for t in ts]
    return xs, ys
end

println("Generating Trajectory Plot...")
try
    x_init, y_init = get_traj(10.0, 0.0)
    x_opt_traj, y_opt_traj = get_traj(v_opt, theta_opt)

    plt1 = plot(title="Projectile Trajectory Optimization", xlabel="Distance (m)", ylabel="Height (m)", legend=:bottomright)
    plot!(plt1, x_init, y_init, label="Initial (v=10, θ=0°)", linestyle=:dash, color=:grey)
    plot!(plt1, x_opt_traj, y_opt_traj, label="Optimized (v=$(round(v_opt,digits=1)), θ=$(round(theta_opt,digits=1))°)", color=:darkblue, lw=2)
    scatter!(plt1, [4.0], [h_hoop], label="Hoop Target", markersize=6, color=:darkgreen, marker=:star5)
    scatter!(plt1, [0.0], [y0_bot], label="Robot Launch", color=:black, markersize=4)

    savefig(plt1, "netball_optimized_en.png")
    println("Saved netball_optimized_en.png")
catch e
    println("Error in Trajectory Plot:")
    showerror(stdout, e)
    println()
end

# IG for Netball
println("--- Running Netball IG ---")
function netball_wrapper(z::Tensor)
    # z is shape 2x1 input
    v = z[1] 
    th = z[2]
    
    theta_rad = th * (3.1415926535 / 180.0)
    term_rise = (d_dist * sin(theta_rad)) * ((cos(theta_rad)) ^ -1)
    denom = (v ^ 2) * (cos(theta_rad) ^ 2)
    num = 0.5 * g_grav * (d_dist ^ 2)
    term_drop = num * (denom ^ -1)
    h_pred = Tensor([y0_bot]) + term_rise + term_drop
    return h_pred
end

x_final = reshape([v_opt, theta_opt], 2, 1)
x_base = reshape([10.0, 0.0], 2, 1)

try
    attrs_netball = integrated_gradients(netball_wrapper, x_final, x_base; steps=50)
    plt2 = bar(["Velocity", "Angle"], vec(attrs_netball), 
        title="Parameter Attribution (Integrated Gradients)", 
        ylabel="Contribution to Height Change (m)", 
        legend=false, 
        color=[:navy, :orange],
        alpha=0.8)
    savefig(plt2, "netball_ig.png")
    println("Saved netball_ig.png")
catch e
    println("Error in Netball IG: ", e)
end

# =========================================================================
# 5. CASE STUDY 2: IRIS CLASSIFICATION
# =========================================================================
println("\n--- Running Iris Classification ---")

# Mini Dataset
X_train = [
    5.1 3.5 1.4 0.2; 4.9 3.0 1.4 0.2; 4.7 3.2 1.3 0.2; 4.6 3.1 1.5 0.2; 5.0 3.6 1.4 0.2; # Setosa
    7.0 3.2 4.7 1.4; 6.4 3.2 4.5 1.5; 6.9 3.1 4.9 1.5; 5.5 2.3 4.0 1.3; 6.5 2.8 4.6 1.5; # Versicolor
    6.3 3.3 6.0 2.5; 5.8 2.7 5.1 1.9; 7.1 3.0 5.9 2.1; 6.3 2.9 5.6 1.8; 6.5 3.0 5.8 2.2  # Virginica
]'

y_train_indices = [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3]

mean_X = mean(X_train, dims=2)
std_X = std(X_train, dims=2)
X_train_norm = (X_train .- mean_X) ./ std_X

struct Linear
    W::Tensor
    b::Tensor
end
function Linear(in_dim, out_dim)
    scale = sqrt(2.0 / (in_dim + out_dim))
    W = Tensor(randn(out_dim, in_dim) .* scale, _op="W")
    b = Tensor(zeros(out_dim, 1), _op="b")
    Linear(W, b)
end
function (l::Linear)(x::Tensor)
    return (l.W * x) + l.b
end

struct MLP
    l1::Linear
    l2::Linear
end
function (m::MLP)(x::Tensor)
    h = relu(m.l1(x))
    out = m.l2(h)
    return out
end

# Initialize model
model = MLP(Linear(4, 8), Linear(8, 3))
lr = 0.05
losses = Float64[]

function cross_entropy(logits::Tensor, target_idx::Int)
    exps = exp(logits)
    out_rg = exps.requires_grad
    sum_exps = Tensor(Base.sum(exps.data), _children=(out_rg ? (exps,) : ()), _op="sum_exp", requires_grad=out_rg)
    if out_rg
        function _backward() exps.grad .+= sum_exps.grad[1] end
        sum_exps._backward = _backward
    end

    log_sum = log(sum_exps)
    target_logit = logits[target_idx] # Uses getindex
    loss = log_sum - target_logit
    return loss
end

println("Training Iris MLP...")
try
    for epoch in 1:100
        total_loss = 0.0
        for i in 1:15
            x_in = Tensor(X_train_norm[:, i:i])
            y_true = y_train_indices[i]
            
            logits = model(x_in)
            loss = cross_entropy(logits, y_true)
            
            total_loss += loss.data[1]
            
            zero_grad!(model.l1.W); zero_grad!(model.l1.b)
            zero_grad!(model.l2.W); zero_grad!(model.l2.b)
            
            backward_iterative(loss)
            
            # Simple SGD
            # Avoid inplace logic if it causes issues with immutable structs (but Tensor data is mutable Array)
            model.l1.W.data .-= lr .* model.l1.W.grad
            model.l1.b.data .-= lr .* model.l1.b.grad
            model.l2.W.data .-= lr .* model.l2.W.grad
            model.l2.b.data .-= lr .* model.l2.b.grad
        end
        push!(losses, total_loss)
    end

    plt3 = plot(losses, label="Training Loss", title="Neural Network Training Curve", xlabel="Epochs", ylabel="Cross-Entropy Loss", lw=2, color=:crimson)
    savefig(plt3, "iris_loss.png")
    println("Saved iris_loss.png")
catch e
    println("Error in Iris Training:")
    showerror(stdout, e)
    println()
end

# IG for Iris (Setosa)
println("--- Running Iris IG ---")
idx = 1 
x_sample = Tensor(X_train_norm[:, idx:idx])
baseline = Tensor(zeros(4,1))
target_class = y_train_indices[idx]

function iris_wrapper(x)
    logits = model(x)
    target_logit = logits[target_class]
    return target_logit
end

try
    attrs_iris = integrated_gradients(iris_wrapper, x_sample.data, baseline.data; steps=50)
    features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    plt4 = bar(features, vec(attrs_iris), 
        title="Feature Importance (Iris Setosa)", 
        ylabel="Attribution Score", 
        legend=false, 
        color=:purple, 
        alpha=0.8)
    savefig(plt4, "iris_ig.png")
    println("Saved iris_ig.png")
catch e
    println("Error in Iris IG: ", e)
end

println("--- Done ---")
