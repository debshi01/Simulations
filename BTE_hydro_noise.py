"""
Electron Hydrodynamic Transport in a Graphene Channel with Current Density Fluctuations and PSD

We solve a simplified hydrodynamic model for electrons in a graphene channel with both 
momentum-relaxing (τ_mr) and momentum-conserving (τ_mc) collisions. The momentum equation is:

    (1/Δt)(u^(n+1) - u^n) - η Δu^(n+1) + (1/τ_mr) u^(n+1) + ∇p^(n+1) = F,
    ∇ · u^(n+1) = 0,

with:
    u      : 2D electron fluid velocity (the current density is proportional to u)
    p      : pressure,
    F      : driving force (e.g. due to an electric field),
    η      : effective viscosity (set proportional to τ_mc),
    τ_mr   : momentum-relaxing collision time,
    τ_mc   : momentum-conserving collision time.

In addition to solving the system via a Chorin projection method, this code:
  - Samples the current density on a fixed grid at every time step.
  - Computes the time-averaged current density and the instantaneous fluctuation field.
  - Extracts the time series at the middle of the channel and computes its power spectral density (PSD),
    so you can see the frequency content of the current density noise.
"""

import fenics as fe
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from fenics import Constant, Point, near

# -------------------------
# Parameters & Physical Data
# -------------------------
L = 4.0         # channel length
H = 1.0         # channel height
Nx, Ny = 81, 21 # mesh resolution

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 100

tau_mr = 0.1   # momentum-relaxing collision time
tau_mc = 0.05  # momentum-conserving collision time

# Effective viscosity proportional to τ_mc
eta = 0.01 * tau_mc

# Driving force: constant force in x-direction
F = Constant((1.0, 0.0))

# -------------------------
# Mesh and Function Spaces
# -------------------------
mesh = fe.RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
velocity_space = fe.VectorFunctionSpace(mesh, "Lagrange", 2)
pressure_space = fe.FunctionSpace(mesh, "Lagrange", 1)

# -------------------------
# Boundary Conditions
# -------------------------
# No-slip on the channel walls (y=0 and y=H)
def channel_walls(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[1], H))

bc_velocity = fe.DirichletBC(velocity_space, Constant((0.0, 0.0)), channel_walls)
velocity_bcs = [bc_velocity]

# -------------------------
# Define Trial/Test Functions and Functions
# -------------------------
u_trial = fe.TrialFunction(velocity_space)
v_test = fe.TestFunction(velocity_space)
p_trial = fe.TrialFunction(pressure_space)
q_test = fe.TestFunction(pressure_space)

# Initial condition: electrons start at rest
u_n = fe.Function(velocity_space)
u_n.assign(Constant((0.0, 0.0)))

# Functions for tentative velocity, corrected velocity, and pressure
u_tent = fe.Function(velocity_space)
u_next = fe.Function(velocity_space)
p_next = fe.Function(pressure_space)

# -------------------------
# Weak Form: Tentative Velocity Step
# -------------------------
a_mom = (1.0/TIME_STEP_LENGTH)*fe.inner(u_trial, v_test)*fe.dx \
        + eta*fe.inner(fe.grad(u_trial), fe.grad(v_test))*fe.dx \
        + (1.0/tau_mr)*fe.inner(u_trial, v_test)*fe.dx
L_mom = (1.0/TIME_STEP_LENGTH)*fe.inner(u_n, v_test)*fe.dx + fe.inner(F, v_test)*fe.dx
A_mom = fe.assemble(a_mom)

# -------------------------
# Weak Form: Pressure Poisson Equation
# -------------------------
a_press = fe.inner(fe.grad(p_trial), fe.grad(q_test))*fe.dx
L_press = - (1.0/TIME_STEP_LENGTH)*fe.div(u_tent)*q_test*fe.dx
A_press = fe.assemble(a_press)

# -------------------------
# Weak Form: Velocity Correction
# -------------------------
a_corr = fe.inner(u_trial, v_test)*fe.dx
L_corr = fe.inner(u_tent, v_test)*fe.dx - TIME_STEP_LENGTH*fe.inner(fe.grad(p_next), v_test)*fe.dx
A_corr = fe.assemble(a_corr)

# -------------------------
# Set up Sampling Grid for Current Density
# -------------------------
n_sample_x, n_sample_y = 100, 50
x_values = np.linspace(0, L, n_sample_x)
y_values = np.linspace(0, H, n_sample_y)
X, Y = np.meshgrid(x_values, y_values)

# Initialize lists to store sampled current density components at each time step
sampled_u_x = []
sampled_u_y = []

# -------------------------
# Time-stepping Loop
# -------------------------
for n in tqdm(range(N_TIME_STEPS)):
    # (1) Solve for tentative velocity u*
    b_mom = fe.assemble(L_mom)
    [bc.apply(A_mom, b_mom) for bc in velocity_bcs]
    fe.solve(A_mom, u_tent.vector(), b_mom, "gmres", "ilu")
    
    # (2) Solve for pressure p^(n+1)
    b_press = fe.assemble(L_press)
    fe.solve(A_press, p_next.vector(), b_press, "gmres", "amg")
    
    # (3) Velocity correction: compute u^(n+1)
    b_corr = fe.assemble(L_corr)
    [bc.apply(A_corr, b_corr) for bc in velocity_bcs]
    fe.solve(A_corr, u_next.vector(), b_corr, "gmres", "ilu")
    
    # Update previous solution
    u_n.assign(u_next)
    
    # Sample the current density (electron velocity) on the defined grid
    u_sample = np.array([u_next(Point(x, y)) for x, y in zip(X.ravel(), Y.ravel())])
    u_sample = u_sample.reshape((n_sample_y, n_sample_x, 2))
    sampled_u_x.append(u_sample[:, :, 0])
    sampled_u_y.append(u_sample[:, :, 1])
    
    # Optionally, plot the instantaneous current density
    fe.plot(u_next)
    plt.title(f"Current Density at t = {(n+1)*TIME_STEP_LENGTH:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pause(0.02)
    plt.clf()

plt.close()

# -------------------------
# Post-processing: Compute Time-Averaged Current Density
# -------------------------
sampled_u_x = np.array(sampled_u_x)  # shape: (N_TIME_STEPS, n_sample_y, n_sample_x)
sampled_u_y = np.array(sampled_u_y)

# Compute time average at each grid point
avg_u_x = np.mean(sampled_u_x, axis=0)
avg_u_y = np.mean(sampled_u_y, axis=0)

# -------------------------
# Compute and Plot Fluctuation Field at Final Time Step
# -------------------------
inst_u_x = sampled_u_x[-1]
inst_u_y = sampled_u_y[-1]

# Compute fluctuation: instantaneous - time average
fluct_u_x = inst_u_x - avg_u_x
fluct_u_y = inst_u_y - avg_u_y

plt.figure(figsize=(8,4))
plt.quiver(X, Y, fluct_u_x, fluct_u_y, scale=1.0)
plt.title("Current Density Fluctuations at Final Time Step")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# -------------------------
# Extract Time Series at a Pixel in the Middle of the Channel for PSD Analysis
# -------------------------
# Choose the middle pixel indices
mid_i = n_sample_y // 2
mid_j = n_sample_x // 2

# Extract the time series for the x-component of the current density at that pixel
signal = sampled_u_x[:, mid_i, mid_j]
# Subtract the mean to obtain the fluctuations (noise)
signal_fluct = signal - np.mean(signal)

# -------------------------
# Compute the Power Spectral Density (PSD)
# -------------------------
# FFT parameters: the time step dt = TIME_STEP_LENGTH
fft_vals = np.fft.fft(signal_fluct)
psd = np.abs(fft_vals)**2
freq = np.fft.fftfreq(N_TIME_STEPS, d=TIME_STEP_LENGTH)

# Only take the positive frequencies for plotting
pos_mask = freq > 0
freq_pos = freq[pos_mask]
psd_pos = psd[pos_mask]

# -------------------------
# Plot the PSD
# -------------------------
plt.figure(figsize=(8,4))
plt.loglog(freq_pos, psd_pos, marker='o', linestyle='-')
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("Power Spectral Density of Current Density Noise (Middle Pixel)")
plt.grid(True, which="both", ls="--")
plt.show()
