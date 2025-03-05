"""
Electron Hydrodynamic Transport in a Graphene Channel using FEniCS

We solve a simplified hydrodynamic model for electrons in a channel where
electron-electron (momentum-conserving) collisions and momentum-relaxing
collisions are both present. The model is:

    (1/Δt)(uⁿ⁺¹ - uⁿ) - η Δuⁿ⁺¹ + (1/τ_mr) uⁿ⁺¹ + ∇pⁿ⁺¹ = F,
    ∇·uⁿ⁺¹ = 0,

with
    u      : 2D electron fluid velocity,
    p      : pressure,
    F      : body force (e.g. due to an electric field driving the electrons),
    η      : shear viscosity (set proportional to τ_mc, the momentum conserving relaxation time),
    τ_mr   : momentum-relaxing collision time,
    τ_mc   : momentum-conserving collision time.

The channel is taken as a rectangle (length L, height H) with no-slip on the top and bottom.
The current density (proportional to u) is plotted at each time step.
"""

import fenics as fe
import matplotlib.pyplot as plt
from tqdm import tqdm
from fenics import near, Constant, Point

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

# Set the effective viscosity proportional to τ_mc (adjust prefactor as needed)
eta = 0.01 * tau_mc

# Driving force: a constant force in x-direction (e.g. representing an electric field)
F = Constant((1.0, 0.0))

# -------------------------
# Mesh and Function Spaces
# -------------------------
mesh = fe.RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)

# Taylor-Hood elements (velocity: degree 2, pressure: degree 1)
velocity_space = fe.VectorFunctionSpace(mesh, "Lagrange", 2)
pressure_space = fe.FunctionSpace(mesh, "Lagrange", 1)

# -------------------------
# Boundary Conditions
# -------------------------
# No-slip on the top and bottom walls (y = 0 and y = H)
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

# Initial condition for velocity: start from rest
u_n = fe.Function(velocity_space)
u_n.assign(Constant((0.0, 0.0)))

# Functions to hold tentative velocity, corrected velocity, and pressure
u_tent = fe.Function(velocity_space)
u_next = fe.Function(velocity_space)
p_next = fe.Function(pressure_space)

# -------------------------
# Weak Form: Tentative Velocity Step
# -------------------------
# Solve for u* from:
#   (1/Δt)(u* - u_n) - η Δu* + (1/τ_mr) u* = F
a_mom = (1.0/TIME_STEP_LENGTH)*fe.inner(u_trial, v_test)*fe.dx \
        + eta*fe.inner(fe.grad(u_trial), fe.grad(v_test))*fe.dx \
        + (1.0/tau_mr)*fe.inner(u_trial, v_test)*fe.dx
L_mom = (1.0/TIME_STEP_LENGTH)*fe.inner(u_n, v_test)*fe.dx + fe.inner(F, v_test)*fe.dx

A_mom = fe.assemble(a_mom)

# -------------------------
# Weak Form: Pressure Poisson Equation
# -------------------------
# Solve for pressure from:
#   <∇p, ∇q> = - (1/Δt) <∇·u*, q>
a_press = fe.inner(fe.grad(p_trial), fe.grad(q_test))*fe.dx
L_press = - (1.0/TIME_STEP_LENGTH)*fe.div(u_tent)*q_test*fe.dx
A_press = fe.assemble(a_press)

# -------------------------
# Weak Form: Velocity Correction
# -------------------------
# Update velocity via:
#   <u_next, v> = <u_tent, v> - Δt <∇p, v>
a_corr = fe.inner(u_trial, v_test)*fe.dx
L_corr = fe.inner(u_tent, v_test)*fe.dx - TIME_STEP_LENGTH*fe.inner(fe.grad(p_next), v_test)*fe.dx
A_corr = fe.assemble(a_corr)

# -------------------------
# Time-stepping Loop
# -------------------------
for n in tqdm(range(N_TIME_STEPS)):
    # (1) Solve for the tentative velocity u*
    b_mom = fe.assemble(L_mom)
    [bc.apply(A_mom, b_mom) for bc in velocity_bcs]
    fe.solve(A_mom, u_tent.vector(), b_mom, "gmres", "ilu")
    
    # (2) Solve for the pressure p^(n+1)
    b_press = fe.assemble(L_press)
    fe.solve(A_press, p_next.vector(), b_press, "gmres", "amg")
    
    # (3) Correct the velocity to enforce incompressibility: u^(n+1)
    b_corr = fe.assemble(L_corr)
    [bc.apply(A_corr, b_corr) for bc in velocity_bcs]
    fe.solve(A_corr, u_next.vector(), b_corr, "gmres", "ilu")
    
    # Update the previous solution
    u_n.assign(u_next)
    
    # Plot current density (here proportional to u_next)
    fe.plot(u_next)
    plt.title(f"Current Density Distribution at t = {(n+1)*TIME_STEP_LENGTH:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pause(0.02)
    plt.clf()

plt.show()
