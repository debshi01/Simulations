"""
Electron Hydrodynamic Transport in a Narrow Graphene Channel with a Metal Disk Inclusion

Domain:
  - Graphene channel: rectangle of length 20 and width 2.
  - Metal disk: circle with diameter 1, centered at (10,1).
  
In graphene: momentum-relaxing collision time τₘᵣ = 1, so the damping term is 1/0.1 = 1.
Inside the metal: τₘᵣ = 0.1, so the damping term is 1/0.1 = 10.
The lower damping in the metal mimics a more diffusive metal region.

Governing equations (using a projection method):
   (1/Δt)(uⁿ⁺¹ – uⁿ) - η Δuⁿ⁺¹ + (1/τₘᵣ(x,y)) uⁿ⁺¹ + ∇pⁿ⁺¹ = F,
   ∇·uⁿ⁺¹ = 0,
where:t
   u      : 2D electron velocity (current density ∝ u)
   p      : pressure
   F      : constant driving force in the x-direction
   η      : effective viscosity (here set proportional to a momentum conserving time τ_mc)
   τₘᵣ(x,y): spatially varying momentum-relaxing time (via 1/τₘᵣ)
"""

import fenics as fe
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from fenics import Constant, Point, near


L = 20.0          
H = 2.0           
Nx, Ny = 161, 21  

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 200


tau_mr_graphene = 1  
tau_mr_metal = 0.1    
tau_mc = 0.01    


eta = 0.28e-3


F = Constant((1.0, 0.0))


mesh = fe.RectangleMesh(Point(0, 0), Point(L, H), Nx, Ny)
velocity_space = fe.VectorFunctionSpace(mesh, "Lagrange", 2)
pressure_space = fe.FunctionSpace(mesh, "Lagrange", 1)


class TauMRCoefficient(fe.UserExpression):
    def eval(self, values, x):
        xc, yc = 10.0, 1.0
        r = 0.5
        if (x[0]-xc)**2 + (x[1]-yc)**2 < r**2:
            values[0] = 1.0 / tau_mr_metal  
        else:
            values[0] = 1.0 / tau_mr_graphene
    def value_shape(self):
        return ()

tau_mr_coeff = TauMRCoefficient(degree=0)

# No-slip on the top and bottom boundaries (y=0 and y=H)
def channel_walls(x, on_boundary):
    return on_boundary and (near(x[1], 0) or near(x[1], H))

bc_velocity = fe.DirichletBC(velocity_space, Constant((0.0, 0.0)), channel_walls)
velocity_bcs = [bc_velocity]


u_trial = fe.TrialFunction(velocity_space)
v_test = fe.TestFunction(velocity_space)
p_trial = fe.TrialFunction(pressure_space)
q_test = fe.TestFunction(pressure_space)


u_n = fe.Function(velocity_space)
u_n.assign(Constant((0.0, 0.0)))


u_tent = fe.Function(velocity_space)
u_next = fe.Function(velocity_space)
p_next = fe.Function(pressure_space)


# Weak Form: Tentative Velocity Step
# (1/Δt)*(u_trial - u_n) - η Δu_trial + (1/τₘᵣ(x,y)) u_trial = F
a_mom = (1.0/TIME_STEP_LENGTH) * fe.inner(u_trial, v_test) * fe.dx \
        + eta * fe.inner(fe.grad(u_trial), fe.grad(v_test)) * fe.dx \
        + tau_mr_coeff * fe.inner(u_trial, v_test) * fe.dx
L_mom = (1.0/TIME_STEP_LENGTH) * fe.inner(u_n, v_test) * fe.dx + fe.inner(F, v_test) * fe.dx # + fe.inner(fe.grad(u_n) * u_n, v_test) * fe.dx

A_mom = fe.assemble(a_mom)


# Weak Form: Pressure Poisson Equation
a_press = fe.inner(fe.grad(p_trial), fe.grad(q_test)) * fe.dx
L_press = - (1.0/TIME_STEP_LENGTH) * fe.div(u_tent) * q_test * fe.dx
A_press = fe.assemble(a_press)


# Weak Form: Velocity Correction
a_corr = fe.inner(u_trial, v_test) * fe.dx
L_corr = fe.inner(u_tent, v_test) * fe.dx - TIME_STEP_LENGTH * fe.inner(fe.grad(p_next), v_test) * fe.dx
A_corr = fe.assemble(a_corr)


n_sample_x, n_sample_y = 100, 20
x_values = np.linspace(0, L, n_sample_x)
y_values = np.linspace(0, H, n_sample_y)
X, Y = np.meshgrid(x_values, y_values)

# Also record the time series of the x-component of the current density at a chosen sample point.
sample_point = Point(L/4, H/2)
time_series = []


for n in tqdm(range(N_TIME_STEPS)):
    b_mom = fe.assemble(L_mom)
    [bc.apply(A_mom, b_mom) for bc in velocity_bcs]
    fe.solve(A_mom, u_tent.vector(), b_mom, "gmres", "ilu")
    
    b_press = fe.assemble(L_press)
    fe.solve(A_press, p_next.vector(), b_press, "gmres", "amg")
    
    b_corr = fe.assemble(L_corr)
    [bc.apply(A_corr, b_corr) for bc in velocity_bcs]
    fe.solve(A_corr, u_next.vector(), b_corr, "gmres", "ilu")
    
    u_n.assign(u_next)
    time_series.append(u_next(sample_point)[0])
    
    fe.plot(u_next)
    plt.title(f"Current Density at t = {(n+1)*TIME_STEP_LENGTH:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.pause(0.02)
    plt.clf()

plt.close()


V0 = fe.FunctionSpace(mesh, "DG", 0)
tau_mr_field = fe.interpolate(tau_mr_coeff, V0)
plt.figure()
p_plot = fe.plot(tau_mr_field, title="Spatial Variation of 1/τₘᵣ")
plt.colorbar(p_plot)
plt.xlabel("x")
plt.ylabel("y")
plt.show()



time_series = np.array(time_series)
signal = time_series - np.mean(time_series)


fft_vals = np.fft.fft(signal)
psd = np.abs(fft_vals)**2
freq = np.fft.fftfreq(N_TIME_STEPS, d=TIME_STEP_LENGTH)


pos_mask = freq > 0
freq_pos = freq[pos_mask]
psd_pos = psd[pos_mask]

plt.figure()
plt.loglog(freq_pos, psd_pos, marker='o', linestyle='-')
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("Power Spectral Density of Current Density Noise (x-component)")
plt.grid(True, which="both", ls="--")
plt.show()
