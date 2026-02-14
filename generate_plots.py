import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists (current dir)
output_dir = "."

print("--- Generating plots with Python ---")

# =========================================================================
# 1. NETBALL OPTIMIZATION
# =========================================================================
print("--- Netball ---")

# Constants
y0_bot = 1.0
h_hoop = 3.1
d_dist = 4.0
g_grav = -9.81

# Model
def netball_forward(v, theta_deg):
    theta_rad = np.radians(theta_deg)
    
    # Rise = d * tan(theta)
    term_rise = d_dist * np.tan(theta_rad)
    
    # Drop = 0.5 * g * d^2 / (v^2 * cos^2(theta))
    denom = (v ** 2) * (np.cos(theta_rad) ** 2)
    num = 0.5 * g_grav * (d_dist ** 2)
    term_drop = num / denom
    
    h_pred = y0_bot + term_rise + term_drop
    return h_pred

def netball_loss(params):
    v, theta = params
    h = netball_forward(v, theta)
    return (h - h_hoop) ** 2

# Derivatives (Manual for optimization)
def netball_grads(params):
    v, theta_deg = params
    
    # Numerical gradient for simplicity and robustness
    eps = 1e-5
    
    l0 = netball_loss([v, theta_deg])
    
    l_v = netball_loss([v + eps, theta_deg])
    grad_v = (l_v - l0) / eps
    
    l_th = netball_loss([v, theta_deg + eps])
    grad_th = (l_th - l0) / eps
    
    return np.array([grad_v, grad_th])

# Optimization
params = np.array([10.0, 0.0]) # v, theta
lr = 0.005 # tuned for python magnitude
history = []

for i in range(2500):
    loss = netball_loss(params)
    grads = netball_grads(params)
    params -= lr * grads
    if i % 100 == 0:
        history.append(loss)

v_opt, theta_opt = params
print(f"Optimized: v={v_opt:.3f}, theta={theta_opt:.3f}")

# Plot Trajectory
def get_traj(v, theta_deg):
    rad = np.radians(theta_deg)
    T = d_dist / (v * np.cos(rad))
    ts = np.linspace(0, T, 100)
    xs = v * np.cos(rad) * ts
    ys = y0_bot + v * np.sin(rad) * ts + 0.5 * g_grav * ts**2
    return xs, ys

xs_init, ys_init = get_traj(10.0, 0.0)
xs_opt, ys_opt = get_traj(v_opt, theta_opt)

plt.figure(figsize=(8, 6))
plt.plot(xs_init, ys_init, 'k--', alpha=0.5, label='Initial (v=10, $\\theta$=0)')
plt.plot(xs_opt, ys_opt, 'r-', linewidth=2, label='Optimized')
plt.scatter([4.0], [h_hoop], c='g', s=100, label='Hoop', zorder=5)
plt.scatter([0.0], [y0_bot], c='b', s=100, label='Robot', zorder=5)
plt.xlim(0, 4.5)
plt.ylim(0, 4.0)
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.title("Netball Trajectory Optimization (Py)")
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig(os.path.join(output_dir, "netball_traj.png"))
plt.close()
print("Saved netball_traj.png")

# IG for Netball
# Function: Reduced Height Diff (h_pred - h_base) attributed to delta_v, delta_theta
# Linear approximation basically since physics is smooth
# IG = (x - x') * Integral(grad)
print("Calculating IG for Netball...")

baseline = np.array([10.0, 0.0])
target = params
steps = 50

total_grads = np.zeros(2)
for i in range(steps):
    alpha = (i + 1) / steps
    interp = baseline + alpha * (target - baseline)
    
    # We need gradient of OUTPUT h_pred w.r.t input
    # grad_h
    eps = 1e-5
    h0 = netball_forward(interp[0], interp[1])
    hv = netball_forward(interp[0] + eps, interp[1])
    hth = netball_forward(interp[0], interp[1] + eps)
    
    gh_v = (hv - h0) / eps
    gh_th = (hth - h0) / eps
    
    total_grads += np.array([gh_v, gh_th])

avg_grads = total_grads / steps
diff = target - baseline
attributions = diff * avg_grads

plt.figure(figsize=(6, 4))
plt.bar(["Velocity", "Angle"], attributions, color=['blue', 'orange'])
plt.title("Connect to Output Height (IG)")
plt.ylabel("Attribution (m)")
plt.savefig(os.path.join(output_dir, "netball_ig.png"))
plt.close()
print("Saved netball_ig.png")


# =========================================================================
# 2. IRIS CLASSIFICATION
# =========================================================================
print("\n--- Iris Classification ---")

# Data (hardcoded subset matching Julia script idea)
# 3 classes, 4 features
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2], [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2],
    [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5],
    [6.3, 3.3, 6.0, 2.5], [5.8, 2.7, 5.1, 1.9], [7.1, 3.0, 5.9, 2.1], [6.3, 2.9, 5.6, 1.8], [6.5, 3.0, 5.8, 2.2]
])
y_train = np.array([0]*5 + [1]*5 + [2]*5)

# Normalize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_norm = (X_train - X_mean) / X_std

# MLP Numpy
np.random.seed(42)
w1 = np.random.randn(4, 8) * np.sqrt(2/12)
b1 = np.zeros(8)
w2 = np.random.randn(8, 3) * np.sqrt(2/11)
b2 = np.zeros(3)

def relu(x): return np.maximum(0, x)
def d_relu(x): return (x > 0).astype(float)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=0) # assuming vector

lr = 0.05
losses = []

# Training Loop (SGD)
print("Training MLP...")
for epoch in range(100):
    total_loss = 0
    for i in range(len(X_train)):
        x = X_norm[i]
        y = y_train[i]
        
        # Forward
        z1 = x @ w1 + b1
        a1 = relu(z1)
        z2 = a1 @ w2 + b2
        
        # Softmax loss grad (logits - onehot)
        # We need prob for loss calc
        exp_z = np.exp(z2 - np.max(z2))
        probs = exp_z / exp_z.sum()
        
        loss = -np.log(probs[y])
        total_loss += loss
        
        # Backward
        # dL/dz2 = probs - one_hot
        dz2 = probs.copy()
        dz2[y] -= 1
        
        dw2 = np.outer(a1, dz2)
        db2 = dz2
        
        da1 = dz2 @ w2.T
        dz1 = da1 * d_relu(z1)
        
        dw1 = np.outer(x, dz1)
        db1 = dz1
        
        # Update
        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2
        
    losses.append(total_loss)

plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.title("Iris Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(os.path.join(output_dir, "iris_loss.png"))
plt.close()
print("Saved iris_loss.png")

# IG for Iris
# Explain 1st sample (Setosa, class 0)
x_sample = X_norm[0]
baseline = np.zeros_like(x_sample)
target_class = 0

print("Calculating IG for Iris...")
# Need gradient of TARGET LOGIT w.r.t INPUT
total_grads = np.zeros_like(x_sample)

for i in range(steps):
    alpha = (i + 1) / steps
    inputs = baseline + alpha * (x_sample - baseline)
    
    # Forward to logit
    z1 = inputs @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    # target logit is z2[0]
    
    # Backward from z2[0]
    dz2 = np.zeros(3)
    dz2[target_class] = 1.0 # Gradient of scalar output (logit) is 1
    
    da1 = dz2 @ w2.T
    dz1 = da1 * d_relu(z1)
    
    # dInput = dz1 @ w1.T
    d_inputs = dz1 @ w1.T
    
    total_grads += d_inputs

avg_grads = total_grads / steps
attributions = (x_sample - baseline) * avg_grads

plt.figure(figsize=(6, 4))
plt.bar(["SL", "SW", "PL", "PW"], attributions, color='purple')
plt.title("Feature Importance (Setosa Sample)")
plt.ylabel("Attribution")
plt.savefig(os.path.join(output_dir, "iris_ig.png"))
plt.close()
print("Saved iris_ig.png")
print("--- Done ---")
