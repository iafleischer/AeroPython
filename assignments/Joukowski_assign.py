import numpy
from potential_flow import *
from matplotlib import pyplot

def Joukowski_Transform(Z, c):
    XI = Z + c/Z
    return XI

def Rotate(Z, z_c, aoa):
    """Given old grid Z, new origin, and aoa, return new coordinates grid: Z_prime
    """
    return (Z - z_c) * numpy.exp(-1j * aoa)

def eval_psi(u_inf, kappa, gamma, z_doublet, z_vortex, Z):
    """Given potential flow config, return psi (scalar matrix)
    """
    x_doublet, y_doublet = z_doublet.real, z_doublet.imag
    x_vortex, y_vortex = z_vortex.real, z_vortex.imag
    # freestream
    psi_freestream = u_inf * Z.imag
    # doublet
    psi_doublet = get_stream_function_doublet(kappa, x_doublet, y_doublet, Z.real, Z.imag)
    # vortex
    psi_vortex = get_stream_function_vortex(gamma, x_vortex, y_vortex, Z.real, Z.imag)
    # superpose psi    
    psi = psi_doublet + psi_freestream + psi_vortex
    return psi

def eval_vel(u_inf, kappa, gamma, z_doublet, z_vortex, Z):
    """Given potential flow config, return velocity complex w
    """
    x_doublet, y_doublet = z_doublet.real, z_doublet.imag
    x_vortex, y_vortex = z_vortex.real, z_vortex.imag
    # freestream velocity components
    u_freestream = u_inf * numpy.ones_like(Z.real)
    v_freestream = numpy.zeros_like(Z.imag)
    # doublet velocity components
    u_doublet, v_doublet = get_velocity_doublet(kappa, x_doublet, y_doublet, Z.real, Z.imag)
    # vortex velocity components
    u_vortex, v_vortex = get_velocity_vortex(gamma, x_vortex, y_vortex, Z.real, Z.imag)
    # superposition
    u = u_freestream + u_doublet + u_vortex
    v = v_freestream + v_doublet + v_vortex
    # z-plane velocity complex
    w = u - 1j*v
    return w

def vel_trans(c, Z, w_z):
    """Given grids, velocity in Z-plane, return velocity in XI-plane
    """
    dxi_dz = 1- (c/Z)**2
    # zeta-plane velocity complex
    w_xi = w_z/dxi_dz
    return w_xi

def eval_cp(w, u_inf):
    """Given the velocity complex w, freestream vel u_inf, return Cp
    """
    return 1.0 - (w * w.conjugate())/u_inf**2

def plot_grid(Z, XI):
    """Z: circle plane grid, XI: airfoil plane grid
    """
    pyplot.figure(figsize=(14, 7))
    pyplot.subplot(1,2,1)
    pyplot.plot(Z.real[0,:], Z.imag[0,:],'-')
    pyplot.scatter(Z.real, Z.imag, s=0.5, c='k')
    pyplot.axis('equal')
    pyplot.title(r'$z$-plane grid')
    pyplot.subplot(1,2,2)
    pyplot.plot(XI.real[0,:], XI.imag[0,:],'-')
    pyplot.scatter(XI.real, XI.imag, s=0.5, c='k')
    pyplot.axis('equal')
    pyplot.title(r'$\xi$-plane grid');

def plot_streamline(Z, XI, psi):
    pyplot.figure(figsize=(14, 7))
    pyplot.subplot(1,2,1)
    pyplot.plot(Z.real[0,:], Z.imag[0,:],'-')
    pyplot.contour(Z.real, Z.imag, psi, 71, colors='k', linestyles='solid')
    pyplot.axis('equal')
    pyplot.title(r'$z$-plane streamline')
    pyplot.subplot(1,2,2)
    pyplot.plot(XI.real[0,:], XI.imag[0,:],'-')
    pyplot.contour(XI.real, XI.imag, psi, 51, colors='k', linestyles='solid')
    pyplot.axis('equal')
    pyplot.title(r'$\xi$-plane streamline');

def plot_vel(Z, XI, w_z, w_xi, step):
    pyplot.figure(figsize=(14, 7))
    pyplot.subplot(1,2,1)
    pyplot.plot(Z.real[0,:], Z.imag[0,:],'-')
    pyplot.quiver(Z.real[::step], Z.imag[::step], w_z.real[::step], -w_z.imag[::step], angles='uv', scale=30)
    pyplot.axis('equal')
    pyplot.xlim(-3.0, 3.0)
    pyplot.ylim(-3.0, 3.0)
    pyplot.title(r'$z$-plane: velocity vector')
    pyplot.subplot(1,2,2)
    pyplot.plot(XI.real[0,:], XI.imag[0,:],'-')
    pyplot.quiver(XI.real[::step], XI.imag[::step], w_xi.real[::step], -w_xi.imag[::step], angles='uv', scale=30)
    pyplot.axis('equal')
    pyplot.xlim(-3.0, 3.0)
    pyplot.ylim(-3.0, 3.0)
    pyplot.title(r'$\xi$-plane: velocity vector');

def plot_cp(Z, XI, cp_z, cp_xi): 
    pyplot.figure(figsize=(14, 7))
    pyplot.subplot(1,2,1)
    #pyplot.plot(Z.real[0,:], Z.imag[0,:],'-')
    pyplot.axis('equal')
    contf1 = pyplot.contourf(Z.real, Z.imag, cp_z.real, levels=numpy.linspace(-1.0, 1.0, 100), extend='both')
    cbar = pyplot.colorbar(contf1)
    cbar.set_ticks(numpy.linspace(-1.0, 1.0, 9))
    pyplot.title(r'$z$-plane $C_p$')
    pyplot.subplot(1,2,2)
    #pyplot.plot(XI.real[0,:], XI.imag[0,:],'-')
    pyplot.axis('equal')
    contf2 = pyplot.contourf(XI.real, XI.imag, cp_xi.real, levels=numpy.linspace(-1.0, 1.0, 100), extend='both')
    cbar = pyplot.colorbar(contf2)
    cbar.set_ticks(numpy.linspace(-1.0, 1.0, 9))
    pyplot.title(r'$\xi$-plane $C_p$');