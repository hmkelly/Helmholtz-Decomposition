"""
@author: Harley Kelly
@email: h.kelly21@imperial.ac.uk

CC BY-NC-SA 4.0
https://creativecommons.org/licenses/by-nc-sa/4.0/
"""

import numpy as np
import multiprocessing

def Helmholtz_Decomp_Poisson(v_field, h):
    """
    This function takes in a 3D field and returns the rotational and irrotational component
    of the v_field using the Helmholtz decomposition in real space (r-space) assuming there
    is no harmonic field. This function should work for any boundary conditions and any field.
    Including internal boundary conditions. It does not include surface terms.
    """
    # Find the divergence of the vfield
    div = calc_divergence(v_field, dx=h, dy=h, dz=h)

    # Find the scalar potential
    scalpot_volume = poisson(div, h)

    scalpot_surface = surface_contribution_scalar(v_field, h)

    scalpot =  scalpot_volume - scalpot_surface

    # Find the irrotational component
    irrotational = - np.asarray(np.gradient((scalpot), h, h, h))

    # Find the curl of the vfield
    curl = calc_curl(v_field, dx=h, dy=h, dz=h)

    # Find the vector potential
    vecpot_volume = np.stack([poisson(curl[0], h),
                       poisson(curl[1], h), 
                       poisson(curl[2], h)])

    vecpot_surface = surface_contribution_vector(v_field, h)

    vecpot = vecpot_volume - vecpot_surface

    # Find the rotational component
    rotational = calc_curl(vecpot, dx=h, dy=h, dz=h)

    return irrotational, rotational

def Helmholtz_Decomp_Poisson_rotational(v_field, h):
    """
    This function takes in a 3D field and returns the rotational component
    of the v_field using the Helmholtz decomposition in real space (r-space) assuming there
    is no harmonic field. This function should work for any boundary conditions and any field.
    Including internal boundary conditions. It does not include surface terms.
    """
    # Find the curl of the vfield
    curl = calc_curl(v_field, dx=h, dy=h, dz=h)

    # Find the vector potential
    vecpot_volume = np.stack([poisson(curl[0], h),
                       poisson(curl[1], h), 
                       poisson(curl[2], h)])

    vecpot_surface = surface_contribution_vector(v_field, h)

    vecpot = vecpot_volume - vecpot_surface

    # Find the rotational component
    rotational = calc_curl(vecpot, dx=h, dy=h, dz=h)

    return rotational

def Helmholtz_Decomp_Poisson_irrotational(v_field, h):
    """
    This function takes in a 3D field and returns the irrotational component
    of the field using the Helmholtz decomposition in real space (r-space) assuming there
    is no harmonic field. This function should work for any boundary conditions and any field.
    Including internal boundary conditions. It does not include surface terms.
    """
    # Find the divergence of the vfield
    div = calc_divergence(v_field, dx=h, dy=h, dz=h)

    # Find the scalar potential
    scalpot_volume = poisson(div, h)

    scalpot_surface = surface_contribution_scalar(v_field, h)

    scalpot =  scalpot_volume - scalpot_surface

    # Find the irrotational component
    irrotational = -np.asarray(np.gradient((scalpot), h, h, h))

    return irrotational

def poisson(field, h):
    """
    A function to calculate the potential from a charge distribution using a Poisson solver
    it is based on the FFT convolution Hockney-Eastwood algorithm.
    ( R. W. Hockney and J. W. Eastwood, Computer simulation using particles (crc Press, 1988).))
    Partially outlined here: https://arxiv.org/pdf/2103.08531
    """
    # Algorithm to 0 pad the field
    xdim, ydim, zdim = field.shape

    # Make a zero array of the correct shape
    padded_field = np.zeros((xdim*2-1,ydim*2-1, zdim*2-1))

    # Place the field into the padded field to give the zeor padded field
    padded_field[0:xdim, 0:ydim, 0:zdim] = field
    
    del field
    
    # Define the gird points
    x, y, z = np.meshgrid(np.arange(0,xdim), np.arange(0,ydim), np.arange(0,zdim), indexing='ij')
    
    # Build the Green's function
    greens = 1 / ( 4 * np.pi * h * np.sqrt( x**2 + y**2 + z**2) )

    # Build the periodic Green's function
    periodic_greens = np.zeros((xdim*2-1,ydim*2-1, zdim*2-1))

    # Place the Greens into the extended domain
    periodic_greens[0:xdim,0:ydim,0:zdim] = greens # left, front, bottom
    periodic_greens[xdim:,0:ydim,0:zdim] = greens[1:,:,:][::-1,:,:] # right, front, bottom
    periodic_greens[0:xdim,ydim:,0:zdim] = greens[:,1:,:][:,::-1,:] # left, back, bottom
    periodic_greens[xdim:,ydim:,0:zdim] = greens[1:,1:,:][::-1,::-1,:] # right, back, bottom
    
    periodic_greens[0:xdim,0:ydim,zdim:] = greens[:,:,1:][:,:,::-1] # left, front, top
    periodic_greens[xdim:,0:ydim,zdim:] = greens[1:,:,1:][::-1,:,::-1] # right, front, top
    periodic_greens[0:xdim,ydim:,zdim:] = greens[:,1:,1:][:,::-1,::-1] # left, back, top
    periodic_greens[xdim:,ydim:,zdim:] = greens[1:,1:,1:][::-1,::-1,::-1] # right, back, top
    
    del greens
    
    # remove the singularity
    periodic_greens[0,0,0] = 1 / ( 4 * h * np.pi)
    
    del x,y,z
    
    # Calculate the FFT of the field and the Green's function
    fft_field = np.fft.fftn(padded_field)
    
    del padded_field
    
    fft_greens = np.fft.fftn(periodic_greens)
    
    del periodic_greens
    
    # Calculate the potential on the padded domain
    padded_potential = np.fft.ifftn(fft_field * fft_greens).real
    
    del fft_field, fft_greens
    
    # Extract the potential from the padded domain
    potential = h * h * h * padded_potential[0:xdim,0:ydim,0:zdim]
    
    return(potential)

def surface_contribution_vector(vec_field, h):

    v_field = vec_field.copy()

    # Algorithm to 0 pad the field
    D, xdim, ydim, zdim = v_field.shape

    # Set everything that isn't a face to 0
    v_field[:,1:xdim-1,1:ydim-1,1:zdim-1] = 0

    # Build a surface normal matrix
    N = np.zeros(v_field.shape)

    # Top face
    N[:,:,:,-1] = np.asarray([0,0,1])[:,np.newaxis,np.newaxis]

    # Bottom face
    N[:,:,:,0] = np.asarray([0,0,-1])[:,np.newaxis,np.newaxis]

    # Left face
    N[:,0,:,:] = np.asarray([-1,0,0])[:,np.newaxis,np.newaxis]

    # Right face
    N[:,-1,:,:] = np.asarray([1,0,0])[:,np.newaxis,np.newaxis]

    # Front face
    N[:,:,0,:] = np.asarray([0,-1,0])[:,np.newaxis,np.newaxis]

    # Back face
    N[:,:,-1,:] = np.asarray([0,1,0])[:,np.newaxis,np.newaxis]

    # Redefine the edges (Remember there are 12 for a cube and n is pointing outwards)

    # Front face bottom
    N[:,:,0,0] = np.asarray([0,-1,-1])[:,np.newaxis] / np.sqrt(2)
    # Front face top
    N[:,:,0,-1] = np.asarray([0,-1,1])[:,np.newaxis] / np.sqrt(2)
    # Front face left
    N[:,0,0,:] = np.asarray([-1,-1,0])[:,np.newaxis] / np.sqrt(2)
    # Front face right
    N[:,-1,0,:] = np.asarray([1,-1,0])[:,np.newaxis] / np.sqrt(2)
    # Back face bottom
    N[:,:,-1,0] = np.asarray([0,1,-1])[:,np.newaxis] / np.sqrt(2)
    # Back face top
    N[:,:,-1,-1] = np.asarray([0,1,1])[:,np.newaxis] / np.sqrt(2)
    # Back face right
    N[:,-1,-1,:] = np.asarray([1,1,0])[:,np.newaxis] / np.sqrt(2)
    # Back face left
    N[:,0,-1,:] = np.asarray([-1,1,0])[:,np.newaxis] / np.sqrt(2)
    # Bottom face right
    N[:,-1,:,0] = np.asarray([1,0,-1])[:,np.newaxis] / np.sqrt(2)
    # Bottom face left
    N[:,-1,:,0] = np.asarray([1,0,-1])[:,np.newaxis] / np.sqrt(2)
    # Top face right
    N[:,-1,:,-1] = np.asarray([1,0,1])[:,np.newaxis] / np.sqrt(2)
    # Top face left
    N[:,0,:,-1] = np.asarray([-1,0,1])[:,np.newaxis] / np.sqrt(2)
    
    # Redefine the vertices (Remember there are 8 for a cube and n is pointing outwards)
    
    # Bottom Front Right
    N[:,-1,0,0] = np.asarray([1,-1,-1]) / np.sqrt(3)

    # Bottom Front Left
    N[:,0,0,0] = np.asarray([-1,-1,-1]) / np.sqrt(3)

    # Bottom Back Right
    N[:,-1,-1,0] = np.asarray([1,1,-1]) / np.sqrt(3)

    # Bottom Back Left
    N[:,0,-1,0] = np.asarray([-1,1,-1]) / np.sqrt(3)

    # Top Front Right
    N[:,-1,0,-1] = np.asarray([1,-1,1]) / np.sqrt(3)

    # Top Front Left
    N[:,0,0,-1] = np.asarray([1,-1,1]) / np.sqrt(3)

    # Top Back Right
    N[:,-1,-1,-1] = np.asarray([1,1,1]) / np.sqrt(3)

    # Top Back Left
    N[:,0,-1,-1] = np.asarray([-1,1,1]) / np.sqrt(3)

    # Cross the face with the normal direction
    field = np.cross(N, v_field, axis=0)

    vecpot_surface = np.stack([poisson(field[0], h),
                       poisson(field[1], h), 
                       poisson(field[2], h)])/h # Divide it by h as the volume has a dV (h^3) where as we want a dS (h^2)
    return(vecpot_surface)

def surface_contribution_scalar(vec_field, h):

    v_field = vec_field.copy()

    # Algorithm to 0 pad the field
    D, xdim, ydim, zdim = v_field.shape

    # Set everything that isn't a face to 0
    v_field[:,1:xdim-1,1:ydim-1,1:zdim-1] = 0

    # Build a surface normal matrix
    N = np.zeros(v_field.shape)

    # Top face
    N[:,:,:,-1] = np.asarray([0,0,1])[:,np.newaxis,np.newaxis]

    # Bottom face
    N[:,:,:,0] = np.asarray([0,0,-1])[:,np.newaxis,np.newaxis]

    # Left face
    N[:,0,:,:] = np.asarray([-1,0,0])[:,np.newaxis,np.newaxis]

    # Right face
    N[:,-1,:,:] = np.asarray([1,0,0])[:,np.newaxis,np.newaxis]

    # Front face
    N[:,:,0,:] = np.asarray([0,-1,0])[:,np.newaxis,np.newaxis]

    # Back face
    N[:,:,-1,:] = np.asarray([0,1,0])[:,np.newaxis,np.newaxis]

    # Redefine the edges (Remember there are 12 for a cube and n is pointing outwards)

    # Front face bottom
    N[:,:,0,0] = np.asarray([0,-1,-1])[:,np.newaxis] / np.sqrt(2)
    # Front face top
    N[:,:,0,-1] = np.asarray([0,-1,1])[:,np.newaxis] / np.sqrt(2)
    # Front face left
    N[:,0,0,:] = np.asarray([-1,-1,0])[:,np.newaxis] / np.sqrt(2)
    # Front face right
    N[:,-1,0,:] = np.asarray([1,-1,0])[:,np.newaxis] / np.sqrt(2)
    # Back face bottom
    N[:,:,-1,0] = np.asarray([0,1,-1])[:,np.newaxis] / np.sqrt(2)
    # Back face top
    N[:,:,-1,-1] = np.asarray([0,1,1])[:,np.newaxis] / np.sqrt(2)
    # Back face right
    N[:,-1,-1,:] = np.asarray([1,1,0])[:,np.newaxis] / np.sqrt(2)
    # Back face left
    N[:,0,-1,:] = np.asarray([-1,1,0])[:,np.newaxis] / np.sqrt(2)
    # Bottom face right
    N[:,-1,:,0] = np.asarray([1,0,-1])[:,np.newaxis] / np.sqrt(2)
    # Bottom face left
    N[:,-1,:,0] = np.asarray([1,0,-1])[:,np.newaxis] / np.sqrt(2)
    # Top face right
    N[:,-1,:,-1] = np.asarray([1,0,1])[:,np.newaxis] / np.sqrt(2)
    # Top face left
    N[:,0,:,-1] = np.asarray([-1,0,1])[:,np.newaxis] / np.sqrt(2)
    
    # Redefine the vertices (Remember there are 8 for a cube and n is pointing outwards)
    
    # Bottom Front Right
    N[:,-1,0,0] = np.asarray([1,-1,-1]) / np.sqrt(3)

    # Bottom Front Left
    N[:,0,0,0] = np.asarray([-1,-1,-1]) / np.sqrt(3)

    # Bottom Back Right
    N[:,-1,-1,0] = np.asarray([1,1,-1]) / np.sqrt(3)

    # Bottom Back Left
    N[:,0,-1,0] = np.asarray([-1,1,-1]) / np.sqrt(3)

    # Top Front Right
    N[:,-1,0,-1] = np.asarray([1,-1,1]) / np.sqrt(3)

    # Top Front Left
    N[:,0,0,-1] = np.asarray([1,-1,1]) / np.sqrt(3)

    # Top Back Right
    N[:,-1,-1,-1] = np.asarray([1,1,1]) / np.sqrt(3)

    # Top Back Left
    N[:,0,-1,-1] = np.asarray([-1,1,1]) / np.sqrt(3)

    # Dot the face with the normal direction to itself
    field = np.einsum('ixyz,ixyz->xyz' ,N, v_field)

    potential = poisson(field,h) / h # Divide it by h as the volume has a dV (h^3) where as we want a dS (h^2)
    return(potential)


def Helmholtz_Decomp_Fourier(F):
    """
    This function takes in a 3D field and returns the irrotational and rotational components
    of the field using the Helmholtz decomposition in Fourier space (k-space) assuming there
    is no harmonic field. This function will not work if the boundary conditions are not periodic or
    Dirichlet (zero values) or if there are sharp discontinuities / things grow too fast or 
    drop off too fast.
    """
    # Define the shape of the array
    nx,ny,nz = F.shape[1:]

    # Define the elements of the field
    Fx, Fy, Fz = F[:]

    # Find the fourier transforms of the field
    Gx = np.fft.fftn(Fx)
    Gy = np.fft.fftn(Fy) 
    Gz = np.fft.fftn(Fz) 

    # Find the values of k
    kx, ky, kz = np.meshgrid(np.fft.fftfreq(nx), np.fft.fftfreq(ny), np.fft.fftfreq(nz), indexing='ij')

    # Find the magnitude of k
    k2 = np.square(kx) + np.square(ky) + np.square(kz)
    k2[0,0,0] = 1. # we do not want an inf. error

    # Calculate the dot product of k and G
    kdotG = kx * Gx + ky * Gy + kz * Gz 

    # Irrotational fourier coefficients
    Gx_Irrotational = np.divide( np.multiply( kdotG, kx ) , k2 ) 
    Gy_Irrotational = np.divide( np.multiply( kdotG, ky ) , k2 ) 
    Gz_Irrotational = np.divide( np.multiply( kdotG, kz ) , k2 ) 
    
    # Rotational fourier coefficients
    Gx_Rotational = Gx - Gx_Irrotational 
    Gy_Rotational = Gy - Gy_Irrotational 
    Gz_Rotational = Gz - Gz_Irrotational 

    # Find the components of F_Irrotational
    Fx_Irrotational = np.fft.ifftn( Gx_Irrotational )
    Fy_Irrotational = np.fft.ifftn( Gy_Irrotational ) 
    Fz_Irrotational = np.fft.ifftn( Gz_Irrotational ) 

    Irrotational = np.asarray( [ Fx_Irrotational , Fy_Irrotational , Fz_Irrotational ] )

    # Find the Rotational component by subtracting Irrotational from orginal F
    Fx_Rotational = np.fft.ifftn( Gx_Rotational )
    Fy_Rotational = np.fft.ifftn( Gy_Rotational )
    Fz_Rotational = np.fft.ifftn( Gz_Rotational )

    Rotational = np.asarray( [ Fx_Rotational , Fy_Rotational , Fz_Rotational ] )

    Irrotational = np.real(Irrotational)
    Rotational = np.real(Rotational)

    return ( Irrotational, Rotational )

def Helmholtz_Decomp_Greens(F, grid, ncores=8):
    """
    This function takes in a 3D field and returns the irrotational and rotational components
    of the field using the Green's function method to calculate the Helmholtz decomposition 
    in real space (r-space) assuming there is no harmonic field. This function is the most 
    accurate and should work for any boundary conditions and any field but the computation 
    time is the slowest. To combat this the code has been made to run in parallel using the 
    multiprocessing module.

    # F is the function being decomposed
    # grid is the grid of points where the field is defined
    # dx, dy, dz are the grid spacings
    # ncore is the number of cores to use
    """
    # Decompose the field into components
    Fx, Fy, Fz = F

    # Extract the dimensions (3D or 2D) and the shape of the box
    dim, x_dim, y_dim, z_dim = grid.shape

    # Find the grid spacings (assumed to be uniform)
    dx = np.diff(grid[0,:,0,0])[0]
    dy = np.diff(grid[1,0,:,0])[0]
    dz = np.diff(grid[2,0,0,:])[0]

    # Find the gradient of F at every point
    dFxdx, dFxdy, dFxdz = np.gradient(Fx, dx, dy, dz)
    dFydx, dFydy, dFydz = np.gradient(Fy, dx, dy, dz)
    dFzdx, dFzdy, dFzdz = np.gradient(Fz, dx, dy, dz)

    del dFxdy, dFxdz, dFydx, dFydz, dFzdx, dFzdy

    # Find the divergence at every point
    div_F = dFxdx + dFydy + dFzdz

    # Set inner boundary contributions to 0
    grid[0,90:150,130:190,130:190] = 0
    grid[1,90:150,130:190,130:190] = 0
    grid[2,90:150,130:190,130:190] = 0

    # Set the field to 0 at the inner boundary
    F[np.where(grid==0)] = 0

    # extract every vector from the grid
    points = grid.reshape(dim, x_dim * y_dim * z_dim).T

    # Define the positions we care calculating at
    if subgrid is None:
        subgrid = grid
        vecs = points
    else:
        sub_dim, x_sub_dim, y_sub_dim, z_sub_dim = subgrid.shape
        vecs = subgrid.reshape(sub_dim, x_sub_dim * y_sub_dim * z_sub_dim).T
    
    # Perform the integrals

    # Define the volume and surface elements
    dV = dx * dy * dz
    dSxy = dx * dy
    dSxz = dx * dz
    dSyz = dy * dz

    # Empty arrays to dump into
    phi_V = []
    phi_S = []

    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=ncores)

    # Calculate phi in parallel for each point
    phi_V, phi_S = np.asarray(pool.starmap(calculate_phi, [(r, dV, dSxy, dSxz, dSyz, div_F, F, grid, dim, x_dim, y_dim, z_dim, points) for r in vecs])).T

    # Make the original arrays
    phi_V = np.asarray(phi_V).reshape(subgrid[0].shape)
    phi_S = np.asarray(phi_S).reshape(subgrid[0].shape)
    
    Irrotational = -np.asarray(np.gradient((phi_V - phi_S), dx, dy, dz))

    # Find the Rotational component by subtracting Irrotational from orginal F
    if F_subgrid is None:
        F_subgrid = F
        
    Rotational = F_subgrid - Irrotational

    return Irrotational, Rotational

def calculate_phi(r, dV, dSxy, dSxz, dSyz, div_F, F, grid, dim, x_dim, y_dim, z_dim, points):
    # Volume integrals
    phi_V = np.sum(np.nan_to_num(dV * div_F.flatten() / ((4 * np.pi) * np.linalg.norm(r - points, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))  # nan to num makes where points=r contribution = 0

    # Surface integrals where n is the outward surface normal
    S = 0

    # outer boundary faces
    # Top face
    n = [0, 0, 1]
    F_srf = F[:, :, :, -1].reshape(dim, x_dim * y_dim) * dSxy
    points_srf = grid[:, :, :, -1].reshape(dim, x_dim * y_dim).T

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))

    # Bottom face
    n = [0, 0, -1]
    F_srf = F[:, :, :, 0].reshape(dim, x_dim * y_dim) * dSxy
    points_srf = grid[:, :, :, 0].reshape(dim, x_dim * y_dim).T

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))

    # Left face
    n = [-1, 0, 0]
    F_srf = F[:, 0, :, :].reshape(dim, y_dim * z_dim) * dSyz
    points_srf = grid[:, 0, :, :].reshape(dim, y_dim * z_dim).T

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))

    # Right face
    n = [1, 0, 0]
    F_srf = F[:, -1, :, :].reshape(dim, y_dim * z_dim) * dSyz
    points_srf = grid[:, -1, :, :].reshape(dim, y_dim * z_dim).T

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))

    # Front face
    n = [0, -1, 0]
    F_srf = F[:, :, 0, :].reshape(dim, x_dim * z_dim) * dSxz
    points_srf = grid[:, :, 0, :].reshape(dim, x_dim * z_dim).T

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))

    # Back face
    n = [0, 1, 0]
    F_srf = F[:, :, -1, :].reshape(dim, x_dim * z_dim) * dSxz
    points_srf = grid[:, :, -1, :].reshape(dim, x_dim * z_dim).T

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))
    phi_S = S / (4 * np.pi)

    S += np.sum(np.nan_to_num(np.einsum('i,iu->iu', n, F_srf) / (np.linalg.norm(r - points_srf, axis=1)), nan=0.0, posinf=0.0, neginf=0.0))
    phi_S = S / (4 * np.pi)

    return phi_V, phi_S


def calc_divergence(v_field, dx=1, dy=1, dz=1):
    """
    Calculate the divergence of a 3D vector field
    """
    grad = np.asarray(np.gradient(v_field, dx, dy, dz, axis=(1,2,3)))
    div = grad[0,0] + grad[1,1] + grad[2,2]
    return div

def calc_curl(v_field, dx=1, dy=1, dz=1):
    """
    Calculate the curl of a 3D vector field
    """
    grad = np.asarray(np.gradient(v_field, dx, dy, dz, axis=(1,2,3)))
    
    curl = np.stack([grad[2,1] - grad[1,2],
                     grad[0,2] - grad[2,0],
                     grad[1,0] - grad[0,1]])
    return curl