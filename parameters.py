import numpy as np

# Group IV-Vacancy Properties
#============================
# Hamiltonian parameters
defect = [f'$^{{29}}$SiV$^–$', f'$^{{73}}$GeV$^–$', f'$^{{117}}$SnV$^–$']
qs = [0.1,0.125,0.15] # [] orbital magnetic field susceptiblity
Ls = [46, 181, 830] # [GHz] spin orbit coupling gnd state
L_excs = [260, 1120, 3000] # [GHz] spin orbit coupling exc state

# Strain susceptibilities
# SiV from experiment (Meesala et al. P.R.B. (2018) https://journals.aps.org/prb/abstract/10.1103/PhysRevB.97.205444)
# GeV assumed same as SiV
# SnV from PBE DFT (Guo et al. arxiv:2307.11916 (2023) https://arxiv.org/abs/2307.11916)
d = [1.3e6, 1.3e6, 0.787e6]     # [GHz/strain]
d_err = [0.1e6, 0.1e6, 0.1e6] # [GHz/strain]
f = [-1.7e6, -1.7e6, -0.562e6]  # [GHz/strain]
f_err = [0.1e6, 0.1e6, 0.1e6]
d_exc = [1.8e6, 1.8e6, 0.956e6]     # [GHz/strain]
d_exc_err = [0.2e6, 0.1e6, 0.1e6] # [GHz/strain]
f_exc = [-3.4e6, -1.7e6, -2.555e6]  # [GHz/strain]
f_exc_err = [0.1e6, 0.1e6, 0.1e6] 

# Strain susceptibility matrices [defect, representation, m, n]
P = [
    np.array([
        [[ d[i]  ,     0  , f[i]/2],
         [    0  , -d[i]  ,    0  ],
         [ f[i]/2,     0  ,    0  ] ],
        [[    0  , -d[i]  ,    0  ],
         [-d[i]  ,     0  , f[i]/2],
         [   0  ,  f[i]/2,    0  ] ]
    ])
for i in range(len(d))]
P = np.array(P)

# Coherence coefficients
chi     = [ 1.81230287e-29, 1.81230287e-29, 6.15725529e-30 ] # [s^2] coherence phonon cross-section parameter
chi_err = np.array(
          [chi,
           [1.62879097e-29, 1.62879097e-29, 4.75762948e-30],
           [2.02865606e-29, 2.02865606e-29, 7.88529044e-30]] # [s^2] coherence phonon cross-section parameter uncertainty range
)

# Group IV-Vacancy Hyperfine Properties
#======================================
# gnd DFT hyperfine for [Si-29, Ge-73, and Sn-117]
rmep = 5.44617021e-4 # ratio of electron to proton mass
rgs = [-1.11058*rmep/2, -0.19544*rmep/2, -2.00208*rmep/2] # Ratio of electron to proton gyromagnetic ratio (assuming g~2 for electrons)
Aisos = np.array([64.2, 48.23, 1389.09]) # [MHz]
Adds = np.array([-2.34, -1.35, -26.65]) # [MHz]
Apars = (Aisos + Adds)/1000 # [GHz]
Aperps = (Aisos - 2*Adds)/1000 # [GHz]
Sns = [1/2, 9/2, 1/2]
S = 1/2

# exc DFT hyperfine [GHz]
Apar_excs = np.array([1.89, 19.33, 672.39])/1000 # [GHz] #(Aiso + Add)/1000 # [GHz]
Aperp_excs = np.array([-46.96, -2.12, 295.82])/1000 # [GHz]#(Aiso - 2*Add)/1000 # [GHz]

# Diamond properties
# ==================
# Diamond tiffness parameters (GPa) at 10 K (Migliori et al. J. Appl. Phys. (2008); https://doi.org/10.1063/1.2975190)
a = 1079.26 #C_11=C_22=C_33
b = 126.73 #C_12=C_13=C_23
c = 578.16 #C_44=C_55=C_66; all others are zero

C_diamond = np.zeros((3,3,3,3))
C_diamond[0,0,0,0] = a
C_diamond[1,1,1,1] = a
C_diamond[2,2,2,2] = a
C_diamond[0,0,1,1] = b
C_diamond[0,0,2,2] = b
C_diamond[1,1,2,2] = b
C_diamond[1,2,1,2] = c/2
C_diamond[2,0,2,0] = c/2
C_diamond[0,1,0,1] = c/2
C_diamond = [[[[ max( C_diamond[i,j,k,l], C_diamond[i,j,l,k], C_diamond[j,i,k,l], C_diamond[j,i,l,k], 
                      C_diamond[k,l,i,j], C_diamond[l,k,i,j], C_diamond[k,l,j,i], C_diamond[l,k,j,i])
                for i in range(3) ] for j in range(3)] for k in range(3)] for l in range(3)] # Enforce index symmetry
C_diamond = np.array(C_diamond) # [GPa] Stiffness tensor in bulk coordinate system X=100, Y=010, Z=001

# Diamond density (ibid)
rho_diamond = 3.501 # g/cm^3

# Electron parameters
# ===================
be = 9.2740100783e-24 # [J/T] # Bohr magneton
ge = 2.0023 # electron g factor
h = 6.626e-34 # [J*s] Plank constant

# Unit converstion
# ================
bohr_to_angstrom = .529177 # [bohr/angstrom]
T_to_GHz = ge*be/h/1e9 # [GHz/T] Conversion from T to GHz
h_GHz = 6.626e-25 # [J/GHz] Plank constant