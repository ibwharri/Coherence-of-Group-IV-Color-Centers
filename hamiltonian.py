import numpy as np

# Hamiltonians
# ============
# Spin matrices formula from: https://easyspin.org/easyspin/documentation/spinoperators.html
Sx = lambda S: 1/2*np.array([ [ np.sqrt((S*(S+1)) - (-S + m)*(-S + n)) if m==(n+1) or (m+1)==n else 0 for n in range(round(2*S+1))] for m in range(round(2*S+1))])
Sy = lambda S: 1/(2*1j)*np.array([ [ (-1)**(m>n)*np.sqrt((S*(S+1)) - (-S + m)*(-S + n)) if m==(n+1) or (m+1)==n else 0 for n in range(round(2*S+1))] for m in range(round(2*S+1))])
Sz = lambda S: np.diag(np.arange(S,-S-0.5,-1))

# Quadrupole moment
Hq = lambda Vxx, Vyy, Vzz, Sn: (   Vxx*np.kron(np.eye(2), np.kron(np.eye(2), np.dot(Sx(Sn),Sx(Sn))))
                                 + Vyy*np.kron(np.eye(2), np.kron(np.eye(2), np.dot(Sy(Sn),Sy(Sn))))
                                 + Vzz*np.kron(np.eye(2), np.kron(np.eye(2), np.dot(Sz(Sn),Sz(Sn)))) )

def create_hamiltonian():
    """Create a Hamiltonian for the group IV color center spin-orbit system"""
    S = 1/2
    X = Sx(S)
    Y = Sy(S)
    Z = Sz(S)
    I = np.eye(round(2*S+1),round(2*S+1))

    # Magnetic field on electron (not no factor of 1/2 needed for electron spin since it is cancelled by g~2)
    Hbxe = lambda bx: bx*np.kron(I, X)
    Hbye = lambda by: by*np.kron(I, Y)
    Hbze = lambda bz: bz*np.kron(I, Z)

    # Magnetic field on orbital degree of freedom
    Hbzo = lambda bz, q: q*bz*np.kron(Z, I)

    # SOC
    Hsoc = lambda L: 2*L*np.kron(Z,Z) # factor of 2 since each Z has a factor of 1/2

    # Strain/JT
    Hegx = lambda alpha: -2*alpha*np.kron(X, I)
    Hegy = lambda  beta:  -2*beta*np.kron(Y, I)

    # dipole moment operators [px, py, pz]
    T = np.array([[-1,-1j],[1,-1j]]).transpose()/np.sqrt(2)
    p = [     # egx/y => eg+/-
        2*Z,  #   Z   =>  -X
        -2*X, #   X   =>  -Y
        I     #   I   =>   I
    ]
    p = [np.einsum('ji,jk,kl->il',T.conj(),x,T) for x in p]
    p = np.array([np.kron(x, I) for x in p]) # index (polarisation, i, j)

    # Total Hamiltonian
    Href = lambda L, alpha, beta: Hsoc(L) + Hegx(alpha) + Hegy(beta) # Reference Hamiltonian including only strain & SOC
    H = lambda bx, by, bz, q, L, alpha, beta: Href(L, alpha, beta) + Hbxe(bx) + Hbye(by) + Hbze(bz) + Hbzo(bz,q) # Total Hamiltonian

    return H, Href, p

def create_hamiltonian_nuclear(Sn):
    """Create a Hamiltonian for the group IV color centers coupled to an intrinsic nucleus with spin Sn"""
    S = 1/2
    X = Sx(S)
    Y = Sy(S)
    Z = Sz(S)
    I = np.eye(round(2*S+1),round(2*S+1))

    Xn = Sx(Sn)
    Yn = Sy(Sn)
    Zn = Sz(Sn)
    In = np.eye(round(2*Sn+1),round(2*Sn+1))

    # Magnetic field on electron (not no factor of 1/2 needed for electron spin since it is cancelled by g~2)
    Hbxe = lambda bx: bx*np.kron(I, np.kron(X, In))
    Hbye = lambda by: by*np.kron(I, np.kron(Y, In))
    Hbze = lambda bz: bz*np.kron(I, np.kron(Z, In))

    # Magnetic field on nucleus (rg =  ratio of nuclear/electron gyromagnetic ratios)
    Hbxn = lambda bx, rg: rg*bx*np.kron(I, np.kron(I, Xn))
    Hbyn = lambda by, rg: rg*by*np.kron(I, np.kron(I, Yn))
    Hbzn = lambda bz, rg: rg*bz*np.kron(I, np.kron(I, Zn))

    # Magnetic field on orbital degree of freedom
    Hbzo = lambda bz, q: q*bz*np.kron(Z, np.kron(I,In))

    # Hyperfine coupling (factor of 1/4 for spin 1/2*spin 1/2)
    Hhf = lambda Aperp, Apar: (Aperp*np.kron(I, np.kron(X,Xn)) + Aperp*np.kron(I, np.kron(Y,Yn)) + Apar*np.kron(I, np.kron(Z,Zn)))

    # Total angular momentum^2 operator
    J2 = (S*(S+1) + Sn*(Sn+1))*np.kron(I, np.kron(I,In)) + 2*(np.kron(I, np.kron(X,Xn)) + np.kron(I, np.kron(Y,Yn)) + np.kron(I, np.kron(Z,Zn)))

    # SOC
    Hsoc = lambda L: 2*L*np.kron(np.kron(Z,Z),In) # factor of 2 since each Z has a factor of 1/2
    Hioc = lambda u: 2*u*np.kron(np.kron(Z,I),Zn) # factor of 2 since each Z has a factor of 1/2

    # Strain/JT
    Hegx = lambda alpha: -2*alpha*np.kron(X, np.kron(I,In))
    Hegy = lambda  beta:  -2*beta*np.kron(Y, np.kron(I,In))

    # dipole moment operators [px, py, pz]
    T = np.array([[-1,-1j],[1,-1j]]).transpose()/np.sqrt(2)
    p = [     # egx/y => eg+/-
        2*Z,  #   Z   =>  -X
        -2*X, #   X   =>  -Y
        I     #   I   =>   I
    ]
    p = [np.einsum('ji,jk,kl->il',T.conj(),x,T) for x in p]
    p = np.array([np.kron(x, np.kron(I,In)) for x in p]) # index (polarisation, i, j)

    # Total Hamiltonian
    Href = lambda L, alpha, beta: Hsoc(L) + Hegx(alpha) + Hegy(beta) # Reference Hamiltonian including only strain & SOC
    H = lambda bx, by, bz, rg, q, Aperp, Apar, L, alpha, beta, upsilon=0: Href(L, alpha, beta) + (Hbxe(bx) + Hbxn(bx,rg)) + (Hbye(by) + Hbyn(by,rg)) + (Hbze(bz) + Hbzn(bz,rg) + Hbzo(bz,q)) + Hhf(Aperp,Apar) + Hioc(upsilon) # Total Hamiltonian

    return H, Href, p, J2

def target_frequency_parameters(f_target, swept_parameter, axis, bx, by, bz, q, L, alpha, beta, interpolate=True):
    """For a given array of group IV parameters that sweeps a parameter (swept_parameter) along a given axis, find the value of swept_parameters that yields the qubit frequency closest to the target frequency. Group IV parameters must be the same shape as the swept_parameter or scalars."""

    H, _, _ = create_hamiltonian()



    Hgiv = H(bx, by, bz, q, L, alpha, beta)
    E, _ = np.linalg.eigh(Hgiv)
    E_Q, _, _ = diagonalised_hamiltonian_parameters( E )

    # Find magnetic field that matches target qubit frequency
    i_B = np.argmin( np.abs(E_Q - f_target), axis = axis )

    if interpolate:
        # Interpolate magnetic field
        dswept_parameter = np.diff(swept_parameter, axis=axis)
        swept_parameter = np.take_along_axis( swept_parameter[...,0,0], np.expand_dims(i_B, axis=axis), axis=axis).squeeze()
        dswept_parameter = np.take_along_axis( dswept_parameter[...,0,0], np.expand_dims(i_B-1, axis=axis), axis=axis).squeeze()
        dE_Q = np.diff(E_Q, axis=axis)
        E_Q = np.take_along_axis( E_Q, np.expand_dims(i_B, axis=axis), axis=axis).squeeze()
        dE_Q = np.take_along_axis( dE_Q, np.expand_dims(i_B-1, axis=axis), axis=axis).squeeze()

        swept_parameter = swept_parameter - np.nan_to_num( dswept_parameter/dE_Q*(E_Q - f_target), posinf=0, neginf=0)

    return swept_parameter

# Calculate Transitions
# =====================
def PLE_transitions(Sn, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc, eta):
    """Calculate PLE transition intensities for a sweep of B-field strength"""
    H, Href, p, J2 = create_hamiltonian_nuclear(Sn)
    Ns = int(2*Sn+1)

    B = np.einsum('k,klm->klm', B, np.ones((B.size, 4*Ns, 4*Ns)))
    bz = B*np.cos(theta)
    bx = B*np.sin(theta)*np.cos(phi)
    by = bx*np.sin(phi)

    # Solve gnd Hamiltonian
    Hplot = H(bx, by, bz, rg, q, Aperp, Apar, L, alpha, beta )
    #Hplot = H(bx, 0, bz, 0, q, 0, 0, L, alpha, 0 ) # nuclear spin removed
    Hplot_ref = Href( L, alpha, 0 )

    E, U = np.linalg.eigh(Hplot) # Calculate eigenvalues
    Eref, _ = np.linalg.eigh(Hplot_ref) # Calculate reference eigenvalues

    alignment = np.einsum('...ji,jk,...kl->...il',U.conj(), J2, U)
    alignment = np.real(np.einsum('...ii->...i', alignment))

    # Solve exc Hamiltonian
    Hplot_exc = H(bx, by, bz, rg, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc  )
    #Hplot_exc = H(bx, 0, bz, 0, q_exc, 0, 0, L_exc, alpha, 0  ) # nuclear spin removed
    Hplot_ref_exc = Href( L_exc, alpha, 0 )

    E_exc, U_exc = np.linalg.eigh(Hplot_exc) # Calculate eigenvalues
    Eref_exc, _ = np.linalg.eigh(Hplot_ref_exc) # Calculate reference eigenvalues

    alignment_exc = np.einsum('...ji,jk,...kl->...il',U_exc.conj(), J2, U_exc)
    alignment_exc = np.real(np.einsum('...ii->...i', alignment_exc))

    # Transition dipole moments
    transition = np.einsum('...ji,mjk,...kl->...mil',U_exc.conj(), p, U)
    transition = np.einsum('...ijk,i->...jk', np.abs(transition)**2, eta)

    return E, Eref, U, alignment, E_exc, Eref_exc, U_exc, alignment_exc, transition

def PLE_spectrum(Sn, intensity, lw, f_meas, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc, eta):
    """Calculate PLE spectrum as sum of gaussians with same linewidth, intensity modulated by transition intensity"""
    
    peak = lambda f, f0, a, sigma: a*(sigma/2)**2/((f-f0)**2 + (sigma/2)**2) # Lorentzian
    Ns = int(2*Sn+1)
    
    E, Eref, _, _, E_exc, Eref_exc, _, _, transition = PLE_transitions(Sn, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, q_exc, Aperp_exc, Apar_exc, L_exc, alpha_exc, beta_exc, eta)

    PLE = [[[ peak( f_meas, (E_exc[j,l] - Eref_exc[0]) - (E[j,k] - Eref[0]), transition[j,l,k], lw) for l in range(2*Ns)] for k in range(2*Ns)] for j in range(B.size)] # indices (b field, excited state index, ground state index, frequency index)
    PLE = intensity*np.array(PLE).sum((1,2))
    return PLE

def ODMR_transitions(Sn, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, eta, Hd = None, upsilon=0):
    """Calculate ODMR transition intensities for a sweep of B-field strength"""
    Ns = int(2*Sn+1)
    H, Href, _, J2 = create_hamiltonian_nuclear(Sn)

    if Hd is None:
        Hbx = H(1,0,0,0,0,0,0,0,0,0)
        Hby = H(0,1,0,0,0,0,0,0,0,0)
        Hbz = H(0,0,1,0,0,0,0,0,0,0)
        p = np.array([ Hbx, Hby, Hbz ])
    else:
        p = np.array(Hd)

    if len(B.shape)==1: B = np.einsum('k,klm->klm', B, np.ones((B.size, 4*Ns, 4*Ns)))
    bz = B*np.cos(theta)
    bx = B*np.sin(theta)*np.cos(phi)
    by = bx*np.sin(phi)

    # Solve gnd Hamiltonian
    Hplot = H(bx, by, bz, rg, q, Aperp, Apar, L, alpha, beta, upsilon )
    #Hplot = H(bx, 0, bz, 0, q, 0, 0, L, alpha, 0 ) # nuclear spin removed
    Hplot_ref = Href( L, alpha, 0 )

    E, U = np.linalg.eigh(Hplot) # Calculate eigenvalues
    Eref, _ = np.linalg.eigh(Hplot_ref) # Calculate reference eigenvalues

    alignment = np.einsum('...ji,jk,...kl->...il',U.conj(), J2, U)
    alignment = np.real(np.einsum('...ii->...i', alignment))

    # Transition dipole moments
    transition = np.einsum('...ji,mjk,...kl->...mil',U.conj(), p, U)
    transition = np.einsum('...ijk,i->...jk', np.abs(transition), eta) # transition is not squared for rabi

    return E, Eref, U, alignment, transition

def ODMR_spectrum(Sn, intensity, lw, f_meas, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, eta, Hd = None, upsilon=0):
    """Calculate ODMR spectrum as sum of gaussians with same linewidth, intensity modulated by transition intensity"""
    
    peak = lambda f, f0, a, sigma: a*(sigma/2)**2/((f-f0)**2 + (sigma/2)**2) # Lorentzian
    Ns = int(2*Sn+1)
    
    E, Eref, _, _, transition = ODMR_transitions(Sn, B, theta, phi, rg, q, Aperp, Apar, L, alpha, beta, eta, Hd, upsilon)

    ODMR = [[[ (l != k)*peak( f_meas, (E[j,l] - Eref[0]) - (E[j,k] - Eref[0]), transition[j,l,k], lw) for l in range(2*Ns)] for k in range(2*Ns)] for j in range(B.size)] # indices (b field, excited state index, ground state index, frequency index)
    ODMR = intensity*np.array(ODMR).sum((1,2))
    return ODMR

transform_x = lambda x, y: x - y
transform_y = lambda x, y, y_shift: x + y - y_shift

def transition_diamond_plot( ax, spec, f, f_lim, N_level, f_exc, col_exc, f_gnd, col_gnd, transition_strength, y_shift, cmap_level, cmap_transition, arrow_kwargs=dict({}), labels=[r'$E_\mathrm{exc}$ (GHz)', r'$E_\mathrm{gnd}$ (GHz)'], allow_equal_transition=True):
    """Plot transition diagram"""
    
    # Plot gnd/exc diamond
    for j in range(N_level):
            ax.plot( transform_x( np.array( [f_exc[j]]*2), np.array( [min(f_gnd), max(f_gnd)] ) ), transform_y(  np.array( [f_exc[j]]*2), np.array( [min(f_gnd), max(f_gnd)] ), y_shift ), '-', color=cmap_level(col_exc[j]), lw=0.75)
            ax.plot( transform_x( np.array( [min(f_exc), max(f_exc)] ), np.array( [f_gnd[j]]*2) ), transform_y( np.array( [min(f_exc), max(f_exc)] ), np.array( [f_gnd[j]]*2), y_shift ), '-', color=cmap_level(col_gnd[j]), lw=0.75)

    # Plot transition levels
    for j in range(N_level):
            for k in range(N_level):
                    if ~allow_equal_transition or j != k:
                        ax.plot( [transform_x( f_exc[j], f_gnd[k] )]*2, [transform_y( f_exc[j], f_gnd[k], y_shift ), 0], '-', color=cmap_transition(transition_strength[j,k]), lw=0.75 )
                        ax.scatter( [transform_x( f_exc[j], f_gnd[k] )]*2, [transform_y( f_exc[j], f_gnd[k], y_shift ) , 0], transition_strength[j,k]/np.max(transition_strength)*5, 'k', zorder=10  )

    # Plot tilted axes
    x_y_ax = transform_x( np.array( [min(f_exc), max(f_exc)*1.5]), np.array( [min(f_gnd), min(f_gnd)] ) )
    y_y_ax = transform_y( np.array( [min(f_exc), max(f_exc)*1.5]), np.array( [min(f_gnd), min(f_gnd)] ), y_shift )
    x_x_ax = transform_x( np.array( [min(f_exc), min(f_exc)]), np.array( [min(f_gnd), max(f_gnd)*1.5] ) )
    y_x_ax = transform_y( np.array( [min(f_exc), min(f_exc)]), np.array( [min(f_gnd), max(f_gnd)*1.5] ), y_shift )

    ax.arrow( x_x_ax[0], y_x_ax[0], np.diff(x_x_ax)[0], np.diff(y_x_ax)[0], color='k', **arrow_kwargs)
    ax.arrow( x_y_ax[0], y_y_ax[0], np.diff(x_y_ax)[0], np.diff(y_y_ax)[0], color='k', **arrow_kwargs)
    rot_alignment_args = {'rotation_mode':'anchor', 'va':'top', 'ha':'center'}
    ax.text( np.mean(x_y_ax) + f_lim/50, np.mean(y_y_ax) - f_lim/50, labels[0], rotation=45, **rot_alignment_args)
    ax.text( np.mean(x_x_ax) - f_lim/50, np.mean(y_x_ax) - f_lim/50, labels[1], rotation=-45, **rot_alignment_args)

    # Plot spectrum
    ax.plot( f, spec, 'navy', lw=0.5)
    ax.spines['bottom'].set_position('zero')

    # Format plot
    ax.set_xlim(-f_lim, f_lim)
    ax.set_ylim(-f_lim-y_shift, f_lim*.5)
    ax.set_yticks([])
    ax.set_xlabel('f (GHz)', loc='left')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)


# Tensor Operations
# =================
def rotate_tensor(T, a, theta=None, order=1):
    """Takes ...x3x... list of 3D tensors, T of rank order, and rotates them around axis defined by a. If rotation angle theta is not defined, angle of rotation is magnitude of a."""

    assert order<14, "Order of tensor is too high for numpy.einsum to handle"
    
    # Find rotation matrix
    if theta is None:
        theta = np.linalg.norm(a)

    a = a/np.linalg.norm(a)

    Rn = rotation_matrix(a, theta)

    # Get subscript
    letters = 'abcdefghijklmnopqrstuvwxyz'
    subscript = [ letters[2*i:2*i+2] + ',' for i in range(order) ]
    subscript.append( '...' + letters[1:2*order+1:2]+'->' )
    subscript.append( '...' + letters[:2*order:2] )
    subscript = "".join(subscript)

    # Get operands
    op = [Rn]*order
    op.append(T)

    # Perform transformation
    T_transformed = np.einsum(subscript, *op)

    return T_transformed

def rotation_matrix(a, theta):
    """Takes a unit vector a, and an angle in radians theta and returns the matrix representation of the rotation around a by the angle theta"""
    Cpm = np.array([[    0,   -a[2],  a[1]],
                    [  a[2],     0  ,-a[0]],
                    [ -a[1],   a[0],    0 ]])

    R = np.cos(theta)*np.eye(3) + np.sin(theta)*Cpm + (1-np.cos(theta))*np.einsum('i,j->ij',a,a)

    return R

def tensor_repeat(*variables_1d):
    """Take one-dimensional arrays and puts them into arrays that all have the same shape, with orders of indicies corresponding to order of variables"""

    # Find shape to reshape variables in to
    shapes = [list(np.array(variable).shape) for variable in variables_1d]
    new_shape = []
    for shape in shapes:
        new_shape += shape
    ndim = len(new_shape)

    # Reshape all variables
    current_dim = 0
    variables_1d = list(variables_1d)
    for i, shape in enumerate(shapes):
        n = len(shape)
        repeat_axis = list(range(current_dim)) + list(range(current_dim+n, ndim))
        variables_1d[i] = np.expand_dims( variables_1d[i], axis=repeat_axis )
        
        for axis in repeat_axis:
            variables_1d[i] = np.repeat( variables_1d[i], new_shape[axis], axis=axis )
        
        current_dim += n

    return tuple(variables_1d)

def kron_array( a, b):
    """Returns Kronecker product of arrays of two matrices while preserving the array. kron_array(a, b)[k0, k1, ..., m0, m1] = a[k0, k1, ..., i0, i1] * a[k0, k1, ..., j0, j1] """

    c = np.array( [[[[ a[...,i0,i1]*b[...,j0,j1] for j1 in range(b.shape[-2]) ] for i1 in range(a.shape[-2]) ] for j0 in range(b.shape[-2]) ] for i0 in range(a.shape[-2]) ] )
    c = c.reshape( (a.shape[-2]*b.shape[-2], a.shape[-1]*b.shape[-1]) + a.shape[:-2] )
    indices = list(range(a.ndim))
    c = c.transpose( indices[2:] + indices[:2] )

    return c

# Coherence Calculations
# ======================
nth = lambda f, T: 1/(np.exp(0.04799 * f/T) - 1) # Thermal phonon occupation; f [GHz], T [K]

thermal_population = lambda E, T: np.einsum( '...i,...->...i', np.exp( - 0.04799 * E/T), 1/np.einsum( '...i->...', np.exp( - 0.04799 * E/T) ) ) # Thermal occupation; E [GHz], T [K]

rhoth = lambda p: np.einsum( '...i,ij->...ij', p, np.eye(p.shape[-1]) ) # Convert thermal occupation to density matrix

rho_qubit = lambda r: np.einsum( '...i,ijk->...jk', r, np.array([Sx(0.5), Sy(0.5), Sz(0.5)])) + np.einsum( '...,jk->...jk', np.ones(r.shape[:-1]), np.eye(2) )/2 # Generate array of arbitrary qubit density matrices defined by polarisation vector r

incoherent_evolution = lambda rho, L: np.einsum('...mnij,...jk,...mnlk->...il', L, rho, L.conj()) - 1/2*( np.einsum('...mnji,...mnjk,...kl->...il', L.conj(), L, rho) + np.einsum('...ij,...mnkj,...mnkl->...il', rho, L.conj(), L) )

T_eff_approx = lambda gamma_0, gamma_S, L: (1-np.exp(-2*np.pi*(gamma_0)/np.abs(L)))/ ( gamma_0 * (1-np.exp(-2*np.pi*(gamma_0+gamma_S)/np.abs(L))) )
T_eff_approx_lorentz = lambda gamma_0, gamma_S, L: 1/( gamma_0 + gamma_S/(1 + (np.pi*gamma_S/np.abs(L))**2) )

def diagonalised_hamiltonian_parameters(E, i=0, j=1):
    """Calculate the qubit and orbital frequencies, and the effective coupling"""
    assert i<j, f"i (={i}) must be less than j (={j})"
    Norb = int( E.shape[-1]/2 )

    omega_Q =    (E[...,j+Norb] - E[...,i+Norb] + E[...,j] - E[...,i])/2
    omega_orb =  (E[...,j+Norb] + E[...,i+Norb] - E[...,j] - E[...,i])/2
    lambda_eff = (E[...,j+Norb] - E[...,i+Norb] - E[...,j] + E[...,i])

    return omega_Q, omega_orb, lambda_eff

def nth_thermal_scattering( E, T, adjust_spontaneous_decay=True):
    """Calculate the thermal population for an array of eigenstates E[...,i] at temperatures T"""

    nths = np.array( [[np.zeros_like(E[...,i]) if i==j else nth( np.abs(E[...,i] - E[...,j]), T) for i in range(E.shape[-1])] for j in range(E.shape[-1])] )
    nths = nths.transpose( list( range(2, nths.ndim) ) + list( range(2) ) )

    if adjust_spontaneous_decay: nths = nths + np.triu( np.ones_like(nths), k=1)

    return nths

def calculate_chiR(P, k, q, c, rho, w):
    """Calculate chi coefficient using the strain susceptibility P, Lebedev points & weights k/w, accoustic phonon displacement modes q, speeds of sound c, and density rho"""
    chi = np.einsum("druv, ku, kvm-> drkm", P, k, q)
    chi = 1/(8*np.pi**2*(rho*1e3)*(6.62607015e-34))*np.einsum( "drkm, km, k ->dr", chi**2, 1/c**5, w)
    
    return chi

def calculate_gamma_reduced( hRij, chi, E ):
    """Calculate decay rate without thermal population (=2*pi*chi_ij*omega^3)"""
    
    shape = hRij.shape[1:]
    gamma_reduced = np.zeros( shape )

    for i in range(shape[-1]):
        for j in range(i+1, shape[-1]):
            gamma_reduced[...,i,j] = 2*np.pi*chi* np.einsum( 'r...,...->...', np.abs(hRij[:,...,i,j])**2, np.abs(2*np.pi*(E[...,i] - E[...,j]))**3)

    shape = list(range(len(shape)))
    gamma_reduced = gamma_reduced + np.transpose(gamma_reduced, axes=shape[:-2] + shape[:-3:-1])

    return gamma_reduced

def find_transition_elements( U ):
    """Find the transition elements hRij from halpa/hbeta in the eigenbasis"""
    Norb = int( U.shape[-1]/2 )

    # Strain matrices in eigenbasis
    halpha = np.array([[ 0, 0, -1, 0], [ 0, 0, 0, -1], [-1, 0, 0, 0], [ 0, -1, 0, 0]])
    halpha = np.kron( halpha, np.eye( int(Norb/2), int(Norb/2) ) )
    halpha = np.einsum('...ji,jk,...kl->...il', U.conj(), halpha, U)
    hbeta  = np.array([[ 0, 0,  1, 0], [ 0, 0, 0,  1], [-1, 0, 0, 0], [ 0, -1, 0, 0]])*1j
    hbeta = np.kron( hbeta, np.eye( int(Norb/2), int(Norb/2) ) )
    hbeta = np.einsum('...ji,jk,...kl->...il',U.conj(), hbeta, U)
    hRij = np.array([halpha, hbeta])

    return hRij

def calculate_gamma_thermalised( E, U, chi, T ):
    """From the energy levels, eigenbasis, phonon cross-sections, and temperature, calculate four-level system decay rates"""
    
    hRij = find_transition_elements( U )

    # Four-level decoherence rates
    shape = hRij.shape[1:]
    gamma = np.zeros( shape )

    for i in range(shape[-1]):
        for j in range(shape[-1]):
            if i > j:
                gamma[...,i,j] = 2*np.pi*chi* np.einsum( 'r...,...,...->...', np.abs(hRij[:,...,i,j])**2, np.abs(2*np.pi*(E[...,i] - E[...,j])*1e9)**3, nth( np.abs((E[...,i] - E[...,j])) , T[...] ) )
            elif i < j:
                gamma[...,i,j] = 2*np.pi*chi* np.einsum( 'r...,...,...->...', np.abs(hRij[:,...,i,j])**2, np.abs(2*np.pi*(E[...,i] - E[...,j])*1e9)**3, nth( np.abs((E[...,i] - E[...,j])) , T[...] ) + 1 )
    
    return gamma

def calculate_orbital_gamma(E, U, chi, T, i=0, j=1):
    """From the energy levels, eigenbasis, phonon cross-sections, and temperature, calculate orbital T1"""
    
    Norb = int( U.shape[-1]/2 )
    hRij = find_transition_elements( U )
    _, E_orb, _ = diagonalised_hamiltonian_parameters( E, i, j )

    gamma_reduced = calculate_gamma_reduced( hRij, chi, E*1e9 )
    gamma = (gamma_reduced[...,i,Norb+i] + gamma_reduced[...,j,Norb+j])/2*(2*nth(E_orb, T[...,0,0]) + 1)

    return gamma

def calculate_gamma(E, U, chi, T, i=0, j=1):
    """From the energy levels, eigenbasis, phonon cross-sections, and temperature, calculate qubit T1, T2 and orbital spin-conserving scattering time for the levels i, j"""

    Norb = int( U.shape[-1]/2 )
    hRij = find_transition_elements( U )

    # Scattering rates
    gamma_reduced = calculate_gamma_reduced( hRij, chi, E*1e9 )

    # Populations
    E_Q, E_orb, _ = diagonalised_hamiltonian_parameters( E, i, j )
    rho_thermal = rhoth( thermal_population(E, T[...,0]) )
    rho_orb_thermal = rho_thermal[..., i::Norb, i::Norb] + rho_thermal[..., j::Norb, j::Norb]

    # Calculate components of qubit coherence times
    gammaQp = ( gamma_reduced[...,i,j] + gamma_reduced[...,Norb+i,Norb+j] )/2*(2*nth(E_Q, T[...,0,0])+1)
    gammaOrbp = ( gamma_reduced[...,i,Norb+j] + gamma_reduced[...,j,Norb+i] )*(nth(E_orb, T[...,0,0]) + rho_orb_thermal[...,1,1])
    gammaOrb = ( gamma_reduced[...,i,Norb+i] + gamma_reduced[...,j,Norb+j] )*(nth(E_orb, T[...,0,0]) + rho_orb_thermal[...,1,1])

    # Calculate coherence from components
    gamma_T1 = gammaQp + gammaOrbp
    gamma_T2 = 0.5*gamma_T1 + 0.5*gammaOrb

    return gamma_T1, gamma_T2, gammaOrb

def gamma_Q( E, L, T, direction='x' ):
    """Calculate the qubit decoherence rate along the x/y/z direction assuming the orbital degree of freedom is thermalised
    Currently extremely slow, so not recommended compared to using calculate_gamma which gives the same result. 
    """

    if direction=='x' or direction==0:
        rhop = rho_qubit( np.array([ 1, 0, 0]))
        rhom = rho_qubit( np.array([-1, 0, 0]))
        sigma = np.kron( np.eye(2), Sx(0.5)*2 )
    elif direction=='y' or direction==1:
        rhop = rho_qubit( np.array([ 0, 1, 0]))
        rhom = rho_qubit( np.array([ 0,-1, 0]))
        sigma = np.kron( np.eye(2), Sy(0.5)*2 )
    if direction=='z' or direction==2:
        rhop = rho_qubit( np.array([ 0, 0, 1]))
        rhom = rho_qubit( np.array([ 0, 0,-1]))
        sigma = np.kron( np.eye(2), Sz(0.5)*2 )

    # Calculate orbital thermal population
    rho_orb_th = rhoth( thermal_population(E, T) )
    rho_orb_th = rho_orb_th[..., ::2, ::2,] + rho_orb_th[..., 1::2, 1::2,]

    # Calculate rate of change of qubit state
    rhop = incoherent_evolution( kron_array( rho_orb_th, rhop ), L )
    rhom = incoherent_evolution( kron_array( rho_orb_th, rhom ), L)
    rp = np.einsum('...ij,ji->...', rhop, sigma)
    rm = np.einsum('...ij,ji->...', rhom, sigma)

    # Calculate relaxation rate and equlilibrium polarisation
    gamma = np.real(rm-rp)/2
    r_eq = np.real(rp+rm)/2
    
    return gamma, r_eq
