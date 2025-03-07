from fenics import *
import numpy as np
import matplotlib.pyplot as plt

Lx = 40e-6     
Ly = 5e-6      
Ltheta = 2*np.pi  
nx, ny, ntheta = 60, 20, 20


v_f = 1e6      
E0 = 1e4        
e_charge = 1.6e-19  
hbar = 1.054e-34    
k_F = 1e9      
F0 = e_charge * E0 / (hbar * k_F)  

tau_mc = 1e-14  
tau_mr = 1e-14  
kappa = 1/tau_mc + 1/tau_mr  

dt = 1e-15  
T_final = 1e-11
num_steps = int(T_final/dt)


class PeriodicTheta(SubDomain):

    def inside(self, x, on_boundary):
         return near(x[2], 0) and on_boundary

    def map(self, x, y):
         y[0] = x[0]
         y[1] = x[1]
         y[2] = x[2] - Ltheta


mesh = BoxMesh(Point(0, 0, 0), Point(Lx, Ly, Ltheta), nx, ny, ntheta)
dx = Measure('dx', domain=mesh)


V = FunctionSpace(mesh, "CG", 1, constrained_domain=PeriodicTheta())

class BoundaryXY(SubDomain):
    def inside(self, x, on_boundary):
         return on_boundary and (near(x[0], 0) or near(x[0], Lx) or near(x[1], 0) or near(x[1], Ly))

bc = DirichletBC(V, Constant(1.0), BoundaryXY())


f = TrialFunction(V)
v = TestFunction(V)
f_n = Function(V)


f_n_vec = f_n.vector().get_local()
noise_amplitude = 0.01
f_n_vec = 1.0 + noise_amplitude * np.random.rand(len(f_n_vec))
f_n.vector()[:] = f_n_vec


cos_theta = Expression("cos(x[2])", degree=2)
sin_theta = Expression("sin(x[2])", degree=2)


#   (f - f_n)/dt + v_f*(cos(θ)∂ₓ f + sin(θ)∂ᵧ f) - F0*sin(θ)*∂_θ f + kappa*f = kappa*1
a = (f/dt)*v*dx \
    + v_f*(cos_theta * f.dx(0) + sin_theta * f.dx(1))*v*dx \
    - F0 * sin_theta * f.dx(2)*v*dx \
    + kappa * f * v * dx

L_form = (f_n/dt)*v*dx + kappa * Constant(1.0)*v*dx

f_new = Function(V)  


nx_plot, ny_plot = 60, 20
x_vals = np.linspace(0, Lx, nx_plot)
y_vals = np.linspace(0, Ly, ny_plot)


n_theta_samples = 50
theta_vals = np.linspace(0, 2*np.pi, n_theta_samples, endpoint=False)
dtheta = 2*np.pi / n_theta_samples

t = 0.0
print("Starting simulation and plotting...")
for n in range(num_steps):
    t += dt
    solve(a == L_form, f_new, bc, solver_parameters={"linear_solver": "gmres", "preconditioner": "ilu"})
    f_n.assign(f_new)


    #   J_x(x,y) = v_f * ∫ cos(θ) f(x,y,θ) dθ.
    Jx = np.zeros((ny_plot, nx_plot))
    for i in range(nx_plot):
        for j in range(ny_plot):
            f_samples = np.array([f_new(Point(x_vals[i], y_vals[j], theta)) for theta in theta_vals])
            Jx[j, i] = v_f * np.sum(f_samples * np.cos(theta_vals)) * dtheta

    if n % 5 == 0:
        fig = plt.figure()
        im = plt.imshow(Jx, origin='lower', extent=[0, Lx, 0, Ly])
        plt.colorbar(im)
        plt.title(f"Current Density J_x at t = {t:.2e} s")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.pause(1)
        plt.close(fig)

print("Simulation complete.")


