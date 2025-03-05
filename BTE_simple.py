"""
Solves a simplified Boltzmann transport (BGK) equation using Finite Elements.
The equation solved is:
    ∂f/∂t + v · ∇f = (f_eq - f) / τ,
where:
    f    : distribution function (scalar)
    f_eq : equilibrium distribution (here constant)
    v    : constant advection velocity
    τ    : relaxation time
This is a simplified kinetic model that mimics transport with relaxation.
We solve the equation on a 2D unit square using FEniCS.
"""

import fenics as fe
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
N_POINTS = 41
TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 300
tau = 0.1                    # Relaxation time
advection_velocity = fe.Constant((1.0, 0.0))  # Constant velocity (moving rightward)
f_eq = fe.Constant(1.0)      # Equilibrium distribution value
f_in = fe.Constant(1.0)      # Inflow boundary condition value

# Create mesh and define function space
mesh = fe.UnitSquareMesh(N_POINTS, N_POINTS)
V = fe.FunctionSpace(mesh, "Lagrange", 1)

# Define trial and test functions
f_trial = fe.TrialFunction(V)
phi = fe.TestFunction(V)

# Define initial condition: f = 0 in the interior
f_n = fe.Function(V)
f_n.assign(fe.Constant(0.0))

# Define inflow boundary (for advection velocity (1,0), inflow is at x=0)
def inflow_boundary(x, on_boundary):
    return on_boundary and fe.near(x[0], 0.0)

bc = fe.DirichletBC(V, f_in, inflow_boundary)

# Time-stepping parameters
dt = TIME_STEP_LENGTH

# Define the weak form
# Left-hand side:
a = f_trial * phi * fe.dx + dt * (
        fe.dot(advection_velocity, fe.grad(f_trial)) * phi * fe.dx +
        (1.0/tau) * f_trial * phi * fe.dx
    )
# Right-hand side:
L = f_n * phi * fe.dx + dt * ((1.0/tau) * f_eq * phi * fe.dx)

# Preassemble the system matrix (time independent)
A = fe.assemble(a)

# Prepare the function to hold the new solution
f = fe.Function(V)

# Time-stepping loop
for n in tqdm(range(N_TIME_STEPS)):
    # Assemble the right-hand side
    b = fe.assemble(L)
    # Apply Dirichlet (inflow) boundary condition
    bc.apply(A, b)
    # Solve the linear system
    fe.solve(A, f.vector(), b, "gmres", "ilu")
    
    # Update the previous solution
    f_n.assign(f)
    
    # Plot the current solution
    fe.plot(f)
    plt.title(f"Boltzmann Transport: Time step {n+1}")
    plt.pause(0.02)
    plt.clf()

plt.show()
