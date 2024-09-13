# HelmholtzDecomposition
A python package for multiple different methods to decompose a vectorfield into its irrotaional and rotaional components.

Poisson based decomposition
A helmholtz decomposition of a vector field based on a poisson solver. This version should work regardless of boundary conditions internal boundary conditions as it is not spatially dependent.

Fourier based decomposition
A helmholtz decomposition of a vector field based on a poisson solver. This version is the fastest and only works for periodic or Dirichlet boudary conditions. A direct cosine transform must be used instead for Neumann conditions. (see https://doi.org/10.3389/fspas.2024.1431238 for a derivation)

Green's based decomposition
A helmholtz decomposition of a vector field based on a poisson solver. This version is the slowlest and but shoudl work irrespective of boundary conditions.

Poisson solver
A function to calculate the potential from a charge distribution using a Poisson solver it is based on the FFT convolution Hockney-Eastwood algorithm. ( R. W. Hockney and J. W. Eastwood, Computer simulation using particles (crc Press, 1988).)
