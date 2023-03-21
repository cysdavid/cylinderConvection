from scipy import special as sp
from scipy.optimize import fsolve
import numpy as np

def cylRayleigh(Γ, Ek, Pr, M=50, K=50, J=25, printVals=True):
    '''
    Returns the minimum (normalized) Rayleigh number and wavenumbers 
    of bulk and wall modes for rotating convection in a cylinder 
    with no-slip boundaries, insulating sidewalls, and isothermal top 
    and bottom. Implements the asymptotic results of Zhang & Liao (2009) 
    (https://doi.org/10.1017/S002211200800517X).

    IMPORTANT: The files 'cylinderRayleigh.py', 'xiRetroGuesses.csv', 
    and 'xiProGuesses.csv' must all exist in the working directory.

    Parameters
    ----------
    Γ : float
        Cylindrical tank aspect ratio.
        Ratio of tank diameter to height
        (Γ = D/H)
    Ek : float
        Ekman number.
        Ratio of viscous to Coriolis forces on the container scale
        (Ek = ν/(Ω H^2))
    Pr : float
        Prandtl number.
        Ratio of viscosity to thermal diffusivity
        (Pr = ν/κ)
    M : int, optional
        Number of azimuthal modes to search. 
        Program will search azimuthal modes m = 1,...,M
        Will not accept M greater than 100.
    K : int, optional
        Number of radial modes to search.
        Program will search radial modes k = 1,...,K
        Will not accept K greater than 100.
    J : int, optional
        A truncation number.
        The number of passive temperature modes used to compute the first-order 
        (viscously-corrected) bulk convection solution.
        Using J less than 25 is not recommended (critical mode may be misidentified).
    printVals : bool, optional
        If True, then values for both bulk and wall onset modes 
        will be printed instead of returned.
        If False, values will be returned instead of printed.

    Returns
    -------
    R1c : float
        Minimum modified Rayleigh number among the first M x K bulk modes.
        Defined as the Rayleigh number scaled by the Ekman number.
        (R = Ra Ek = α ΔT g H/(Ω κ))
    σ1c : float
        Dimensionless half-frequency of the bulk mode with R = R1c,
        scaled by the rotation rate, Ω.
    m0c : float
        Azimuthal wavenumber of the bulk mode with R = R1c.
    k0c : float
        Radial wavenumber of the bulk mode with R = R1c.
    Rwall : float
        Minimum modified Rayleigh number among sidewall-localized (viscous) convective modes. 
        Rescaled from the asymptotic solution in a channel.
    σwall : float
        Dimensionless half-frequency of the wall mode with R = Rwall
        Rescaled from the asymptotic solution in a channel.
    mwall : float
        Azimuthal wavenumber of the wall mode with R = Rwall
        Rescaled from the asymptotic solution in a channel.

    Notes
    -----
    Symbols used in documentation:
        m  - azimuthal wavenumber
        n  - vertical wavenumber
        k  - radial wavenumber
        R  - tank radius
        D  - tank diameter
        H  - tank height
        ν  - fluid viscosity
        Ω  - rotation rate
        κ  - thermal diffusivity
        α  - thermal expansion coefficient
        ΔT - top to bottom temperature difference
        g  - gravitation acceleration
        R  - modified Rayleigh number

    For both bulk and wall onset modes, the simplest vertical structure is assumed (n = 1).

    Applying the eigenvalue condition (equation 4.10 in Zhang & Liao, 2009) on radial 
    wavenumber ξ requires numerical root-finding. This program uses pre-computed 
    roots of (4.10) with  Γ = 2, M = 100, K = 100 as starting guesses for finding 
    roots with Γ != 2. The starting guesses are stored in the files 'xiRetroGuesses.csv'
    and 'xiProGuesses.csv'

    References
    -----
    Zhang, K., & Liao, X. (2009). The onset of convection in rotating circular cylinders 
        with experimental boundary conditions. Journal of Fluid Mechanics, 622, 63-73. 
        doi:10.1017/S002211200800517X

    Zhang, K., & Liao, X. (2017). Theory and Modeling of Rotating Fluids: Convection,
        Inertial Waves and Precession (Cambridge Monographs on Mechanics). Cambridge: 
        Cambridge University Press. doi:10.1017/9781139024853
    '''
    # Convert Γ = D/H to γ = r/H
    γ = Γ/2

    # Find the jth smallest positive root $\beta_{m1j}$ of $J_m'(\beta) = 0$ for each $m$, $1\le m \le M$.

    β = np.zeros((M,K,J))
    for m in range(1,M+1):
        for k in range(1,K+1):
            β[m-1,k-1,:] = sp.jnp_zeros(m, J)
    del m
    del k

    # Solve transcendental equation for $\xi$ using the roots for the γ = 1 case as starting points:

    ξRetroGuesses = np.loadtxt('xiRetroGuesses.csv',delimiter=',')[:M,:K]
    ξProGuesses = np.loadtxt('xiProGuesses.csv',delimiter=',')[:M,:K]

    def ξRetroFunc(ξ,m,γ):
        ξEqnp = ξ*sp.jv(-1 + m,ξ) + m*(-1 + np.sqrt(1 +
                ξ**2/(np.pi**2*γ**2)))*sp.jv(m,ξ)
        return ξEqnp

    def ξProFunc(ξ,m,γ):
        ξEqnp = ξ*sp.jv(-1 + m,ξ) + m*(-1 - np.sqrt(1 +
                ξ**2/(np.pi**2*γ**2)))*sp.jv(m,ξ)
        
        return ξEqnp

    ξRetro = np.zeros((M,K))
    ξPro = np.zeros((M,K))

    if γ < 0.2:
        import copy
        
        ξRetroGuesses0 = copy.deepcopy(ξRetroGuesses)
        ξProGuesses0 = copy.deepcopy(ξProGuesses)
        
        for m in range(1,M+1):
            fRetro = lambda ξ : ξRetroFunc(ξ,m,0.2)
            ξRetroGuesses[m-1,:] = fsolve(fRetro,ξRetroGuesses0[m-1,:])

            fPro = lambda ξ : ξProFunc(ξ,m,0.2)
            ξProGuesses[m-1,:] =fsolve(fPro,ξProGuesses0[m-1,:])
            
    for m in range(1,M+1):
        fRetro = lambda ξ : ξRetroFunc(ξ,m,γ)
        ξRetro[m-1,:] = fsolve(fRetro,ξRetroGuesses[m-1,:K])
        
        fPro = lambda ξ : ξProFunc(ξ,m,γ)
        ξPro[m-1,:] =fsolve(fPro,ξProGuesses[m-1,:K])
        
    warningCond = np.sum(np.abs(ξPro[:M,:K-1]-ξProGuesses[:M,:K-1]) > np.abs(np.diff(ξProGuesses[:M,:K],axis=1)))!=0

    if warningCond:
        print("Warning: some roots may not have been found.")
        
    # Use the dispersion relation to compute the corresponding inviscid half-frequencies $\sigma_0$

    σ0Retro = (1 + (ξRetro/(γ*np.pi))**2)**(-1/2)
    σ0Pro = -(1 + (ξPro/(γ*np.pi))**2)**(-1/2)

    ξ = np.stack((ξRetro,ξPro))
    σ0 = np.stack((σ0Retro,σ0Pro))

    mArr = np.array([[m for k in range(1,K+1)] for m in range(1,M+1)])
    kArr = np.array([[k for k in range(1,K+1)] for m in range(1,M+1)])

    # Compute R1 for every mode m,k ($1\le m \le M$, $1\le k \le K$).

    Sum = np.zeros((2,M,K))
    Sum2 = np.zeros((2,M,K))

    for j in range(1,J+1):
        Q = (np.pi**2*γ**2*β[...,j-1]**2)/(2*(-mArr**2 + β[...,j-1]**2)*(-β[...,j-1]**2+ ξ**2)**2*σ0**2)
        Sum += (Q*(np.pi**2 + β[...,j-1]**2/γ**2))/((4*Pr**2*σ0**2)/Ek**2 + (np.pi**2 + β[...,j-1]**2/γ**2)**2)
        Sum2 += Q/((np.pi**2 + β[...,j-1]**2/γ**2)**2 + (4*Pr**2*σ0**2)/Ek**2)

    with np.errstate(divide='ignore'):
        R1 = (np.sqrt(Ek)*((np.sqrt(Ek)*np.pi**2*(np.pi**2*γ**2 
            + mArr*(mArr - σ0)))/(σ0**2*np.sqrt(1 - σ0**2)) 
            - 2*mArr*σ0*(np.sqrt(1 - σ0) + np.sqrt(1 + σ0)) 
            + (mArr**2 + np.pi**2*γ**2)*((1 - σ0)**1.5 + (1 + σ0)**1.5 
            + (np.sqrt(1 - σ0**2)*np.sqrt(np.abs(σ0)))/γ)))/(4.*mArr**2*Sum*np.sqrt(1 - σ0**2))

    # Compute minimum R1 and corresponding inviscid half-frequency and wavenumbers.

    c0ind = np.unravel_index(np.argmin(R1, axis=None), R1.shape)
    σ0c = σ0[c0ind]
    ξ0c = ξ[c0ind]
    R1c = R1[c0ind]
    k0c = c0ind[2]+1
    m0c = c0ind[1]+1

    # Compute the corresponding viscously-corrected half-frequency

    σ1c = 0.5*(2*σ0c - 1/(np.pi**2*γ**2 + m0c*(m0c - σ0c))*(np.sqrt(1
            - σ0c**2)*np.sqrt(Ek))*(Sum2[c0ind]*(8*m0c**2*R1c*Pr*σ0c*np.sqrt(1 -
            σ0c**2))/Ek**1.5 + 2*σ0c*m0c*(np.sqrt(1 + σ0c) - np.sqrt(1 -
            σ0c)) + (m0c**2 + np.pi**2*γ**2)*((σ0c*np.sqrt(1 -
            σ0c**2))/(γ*np.sqrt(np.abs(σ0c))) - (1 + σ0c)**1.5 +
            (1 - σ0c)**1.5)))

    # Compute critical Rayleigh number, half-frequency, and azimuthal mode number for viscous (wall) convection

    Rwall = 2*np.sqrt((6*(9 + 5*np.sqrt(3)))/(5 + 3*np.sqrt(3)))*np.pi**2 + 73.8*Ek**(1/3)
    σwall = (-290.6*Ek**(4/3))/Pr + (np.sqrt(2 + np.sqrt(3))*(3 + np.sqrt(3))*Ek*np.pi**2)/((1 + np.sqrt(3))*Pr)
    mwall = γ*(np.sqrt(2 + np.sqrt(3))*np.pi - 27.76*Ek**(1/3))

    if printVals==True:
         
        print(f"Bulk: Rc = {R1c:.4f}, σc =  {σ1c:.4f}, mc = {m0c}, nc = 1, kc = {k0c}")
        print(f"Wall: Rc = {Rwall:.4f}, σc =  {σwall:.4f}, mc = {mwall:.4f}, nc = 1")
    
    else:
        return R1c, σ1c, m0c, k0c, Rwall, σwall, mwall