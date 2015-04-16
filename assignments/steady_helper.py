import numpy
from scipy import integrate


def integral(x, y, panel, dxdz, dydz):
    """Return the integral, dz is the direction. x,y are the evaluated point
    """
    def func(s):
        return ( ((x - (panel.xa - numpy.sin(panel.beta)*s))*dxdz 
                  + (y - (panel.ya + numpy.cos(panel.beta)*s))*dydz)
                / ((x - (panel.xa - numpy.sin(panel.beta)*s))**2 
                   + (y - (panel.ya + numpy.cos(panel.beta)*s))**2) )
    return integrate.quad(lambda s:func(s), 0., panel.length)[0]



def source_matrix(panels):
    A = numpy.empty((panels.size, panels.size), dtype=float)
    numpy.fill_diagonal(A, 0.5)
    
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                A[i,j] = 0.5/numpy.pi*integral(p_i.xc, p_i.yc, p_j, numpy.cos(p_i.beta), numpy.sin(p_i.beta))
    
    return A


def vortex_array(panels):
    a = numpy.zeros(panels.size, dtype=float)
    
    for i, p_i in enumerate(panels):
        for j, p_j in enumerate(panels):
            if i != j:
                a[i] -= 0.5/numpy.pi*integral(p_i.xc, p_i.yc, p_j, numpy.sin(p_i.beta), -numpy.cos(p_i.beta))
    return a


def kutta_array(panels):
    """Builds the Kutta-condition array.
    return a -- 1D array (Nx1, N is the number of panels).
    """
    N = panels.size
    a = numpy.zeros(N+1, dtype=float)
    # contribution from the source sheet of the first panel on the last one
    a[0] = 0.5/numpy.pi*integral(panels[N-1].xc, panels[N-1].yc, panels[0], 
                           -numpy.sin(panels[N-1].beta), +numpy.cos(panels[N-1].beta))
    # contribution from the source sheet of the last panel on the first one
    a[N-1] = 0.5/numpy.pi*integral(panels[0].xc, panels[0].yc, panels[N-1], 
                             -numpy.sin(panels[0].beta), +numpy.cos(panels[0].beta))
    # contribution from the vortex sheet of the first panel on the last one
    a[N] -= 0.5/numpy.pi*integral(panels[-1].xc, panels[-1].yc, panels[0], 
                               +numpy.cos(panels[-1].beta), numpy.sin(panels[-1].beta))
    # contribution from the vortex sheet of the last panel on the first one
    a[N] -= 0.5/numpy.pi*integral(panels[0].xc, panels[0].yc, panels[-1], 
                               +numpy.cos(panels[0].beta), numpy.sin(panels[0].beta))
    # contribution from the vortex sheet of the first panel on itself
    a[N] -= 0.5
    # contribution from the vortex sheet of the last panel on itself
    a[N] -= 0.5
 
    # contribution from the other panels on the first and last ones
    for i, panel in enumerate(panels[1:-1]):
        # contribution from the source sheet
        a[i+1] = 0.5/numpy.pi*(integral(panels[0].xc, panels[0].yc, panel, 
                               -numpy.sin(panels[0].beta), +numpy.cos(panels[0].beta))
                     + integral(panels[N-1].xc, panels[N-1].yc, panel, 
                               -numpy.sin(panels[N-1].beta), +numpy.cos(panels[N-1].beta)) )

        # contribution from the vortex sheet
        a[N] -= 0.5/numpy.pi*(integral(panels[0].xc, panels[0].yc, panel, 
                               +numpy.cos(panels[0].beta), numpy.sin(panels[0].beta))
                             + integral(panels[-1].xc, panels[-1].yc, panel, 
                               +numpy.cos(panels[-1].beta), numpy.sin(panels[-1].beta)) )
        
    return a


def build_matrix(panels):
    N = len(panels)
    A = numpy.empty((N+1, N+1), dtype=float)
    
    AS = source_matrix(panels)
    av = vortex_array(panels)
    ak = kutta_array(panels)
    
    A[0:N,0:N], A[0:N,N], A[N,:] = AS[:,:], av[:], ak[:]
    
    return A



def build_rhs(panels, freestream):
    N = len(panels)
    b = numpy.empty(N+1,dtype=float)
    
    for i, panel in enumerate(panels):
        b[i] = - freestream.u_inf * numpy.cos(freestream.alpha - panel.beta)
    b[N] = -freestream.u_inf*( numpy.sin(freestream.alpha-panels[0].beta)
                              +numpy.sin(freestream.alpha-panels[N-1].beta) )
    
    return b



def solve(panels, freestream):

	A = build_matrix(panels)
	b = build_rhs(panels, freestream)
	solution = numpy.linalg.solve(A, b)

	return solution
