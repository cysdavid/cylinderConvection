from scipy import special as sp
from scipy.optimize import fsolve
import numpy as np
import os
import copy

class cylConvection():
    '''
    Class for computing the minimum normalized Rayleigh number and 
    wavenumbers of oscillatory and wall modes for rotating convection in a 
    cylinder with no-slip boundaries, insulating sidewalls, and isothermal 
    top and bottom. Implements the asymptotic results of Zhang & Liao (2009) 
    (https://doi.org/10.1017/S002211200800517X).

    IMPORTANT: The files 'cylinderConvection.py', 'xiRetroGuesses.csv', 
    and 'xiProGuesses.csv' must exist in the following file structure
    with respect to the working directory:
    
    working directory
    │
    ├── cylinderConvection
    │   │
    │   ├─── __init__.py
    │   ├─── cylinderConvection.py
    │   ├─── proRetroGuesses.csv
    │   |___ xiRetroGuesses.csv
    │
    └── your script
    
    '''

    # Import roots ξ for γ = 1 to use as guesses
    pkgdir = 'cylinderConvection'
    pathRetro = os.path.join(pkgdir,'xiRetroGuesses.csv')
    pathPro = os.path.join(pkgdir,'xiProGuesses.csv')
    
    try:
        ξRetroGuesses = np.loadtxt(pathRetro,delimiter=',')
        ξProGuesses = np.loadtxt(pathPro,delimiter=',')
    except OSError as e:
        raise FileNotFoundError("move xiRetroGuesses.csv and xiProGuesses.csv to package directory")

    def __init__(self, Γ, M, K, J):
        '''
        Initializes cylConvection. Computes roots ξ and β for
        specified Γ, M, K, J.

        Parameters
        ----------
        Γ : float
            Cylindrical tank aspect ratio.
            Ratio of tank diameter to height
            (Γ = D/H)
        M : int
            Number of azimuthal modes to search. 
            Program will search azimuthal modes m = 1,...,M
            Will not accept M greater than 90.
        K : int
            Number of radial modes to search.
            Program will search radial modes k = 1,...,K
            Will not accept K greater than 90.
        J : int
            A truncation number.
            The number of passive temperature modes used to compute the 
            first-order (viscously-corrected) bulk convection solution.
            J = O(M) = O(K) is recommended.

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

        Applying the eigenvalue condition (equation 4.10 in Zhang & Liao, 2009) on radial 
        wavenumber ξ requires numerical root-finding. This program uses pre-computed 
        roots of (4.10) with  Γ = 2, M = 100, K = 100 as starting guesses for finding 
        roots with Γ != 2. The starting guesses are stored in the files 'xiRetroGuesses.csv'
        and 'xiProGuesses.csv'

        References
        ----------
        Zhang, K., & Liao, X. (2009). The onset of convection in rotating circular cylinders 
            with experimental boundary conditions. Journal of Fluid Mechanics, 622, 63-73. 
            doi:10.1017/S002211200800517X

        Zhang, K., & Liao, X. (2017). Theory and Modeling of Rotating Fluids: Convection,
            Inertial Waves and Precession (Cambridge Monographs on Mechanics). Cambridge: 
            Cambridge University Press. doi:10.1017/9781139024853
            
        '''

        if M > 90:
            raise ValueError('M cannot be greater than 90.')
        if K > 90:
            raise ValueError('K cannot be greater than 90.')

        self.Γ = Γ
        self.M = M
        self.K = K
        self.J = J

        # Convert Γ = D/H to γ = r/H
        self.γ = self.Γ/2

        # Find the jth smallest positive root β_{m1j} of J_m'(β) = 0 for each m, 1 <= m <= M.

        self.β = np.zeros((self.M,self.K,self.J))
        for m in range(1,self.M+1):
            for k in range(1,self.K+1):
                self.β[m-1,k-1,:] = sp.jnp_zeros(m, self.J)

        # Solve transcendental equation for ξ using the roots for the γ = 1 case as starting points:

        self.ξRetro = np.zeros((M,K))
        self.ξPro = np.zeros((M,K))

        self.ξRetroGuesses0 = copy.deepcopy(cylConvection.ξRetroGuesses)[:self.M,:]
        self.ξProGuesses0 = copy.deepcopy(cylConvection.ξProGuesses)[:self.M,:]

        if self.γ < 0.2:
            # Solve transcendental equation for ξ with γ = 0.2. 
            # These roots will be used as starting guesses to find the roots with γ < 0.2
            
            for m in range(1,M+1):
                fRetro = lambda ξ : cylConvection.ξRetroFunc(ξ,m,0.2)
                self.ξRetroGuesses0[m-1,:] = fsolve(fRetro,cylConvection.ξRetroGuesses[m-1,:])

                fPro = lambda ξ : cylConvection.ξProFunc(ξ,m,0.2)
                self.ξProGuesses0[m-1,:] = fsolve(fPro,cylConvection.ξProGuesses[m-1,:])

        for m in range(1,M+1):
            # Retrograde roots

            fRetro = lambda ξ : cylConvection.ξRetroFunc(ξ,m,self.γ)
            fsolveResultsRetro = fsolve(fRetro,self.ξRetroGuesses0[m-1,:])
            
            # Eliminate repeated retrograde roots 
            # Roots closer than 1% of the average distance between roots count as repeated
            
            ΔξRetro = np.mean(np.diff(fsolveResultsRetro))/np.mean(fsolveResultsRetro)
            decRetro = abs(int(np.log10(0.01*ΔξRetro)))
            _,retroUniqInds = np.unique(np.round(fsolveResultsRetro,decRetro),return_index=True)

            if len(self.ξRetro[m-1,:]) != len(fsolveResultsRetro[retroUniqInds][:self.K]):
                dKRetro = self.K - len(fsolveResultsRetro[retroUniqInds][:self.K])
                msgRetro = f'Less than K unique retrograde roots at m = {m:.0f}. Decrease K by {dKRetro:.0f}, at least'
                raise ValueError(msgRetro)
            else:
                
                self.ξRetro[m-1,:] = fsolveResultsRetro[retroUniqInds][:self.K]

            # Prograde roots

            fPro = lambda ξ : cylConvection.ξProFunc(ξ,m,self.γ)
            fsolveResultsPro = fsolve(fPro,self.ξProGuesses[m-1,:])
            
            # Eliminate repeated prograde roots 
            # Roots closer than 1% of the average distance between roots count as repeated
            
            ΔξPro = np.mean(np.diff(fsolveResultsPro))/np.mean(fsolveResultsPro) 
            decPro = abs(int(np.log10(0.01*ΔξPro)))
            _,proUniqInds = np.unique(np.round(fsolveResultsPro,decPro),return_index=True)
            self.ξPro[m-1,:] = fsolveResultsPro[proUniqInds][:self.K]
            if len(self.ξPro[m-1,:]) != len(fsolveResultsPro[proUniqInds][:self.K]):
                dKPro = self.K - len(fsolveResultsPro[proUniqInds][:self.K])
                msgPro = f'Less than K unique prograde roots at m = {m:.0f}. Decrease K by {dKPro:.0f}, at least'
                raise ValueError(msgPro)
            else:
                self.ξPro[m-1,:] = fsolveResultsPro[proUniqInds][:self.K]

        # Check roots via graphical method

        for mplot in range(1,self.M+1):
            # Get average distance between roots, Δξ
            ΔξRetroG = np.mean(np.diff(self.ξRetro[mplot-1,:]))
            ΔξProG = np.mean(np.diff(self.ξPro[mplot-1,:]))
            
            # Set grid spacing as gridres*Δξ
            gridres = 0.01
            ξRetroPlotarr = np.arange(self.ξRetro[mplot-1,0]-0.1*ΔξRetroG,self.ξRetro[mplot-1,-1]+0.1*ΔξRetroG,gridres*ΔξRetroG)
            ξProPlotarr = np.arange(self.ξPro[mplot-1,0]-0.1*ΔξProG,self.ξPro[mplot-1,-1]+0.1*ΔξProG,gridres*ΔξProG)
            
            # Assume that the function changes sign at each root and use this to graphically estimate roots
            ξRetroPlot = cylConvection.ξRetroFunc(ξRetroPlotarr,mplot,self.γ)
            ξProPlot = cylConvection.ξProFunc(ξProPlotarr,mplot,self.γ)

            ξRetroMask = np.zeros(len(ξRetroPlotarr))
            ξRetroMask[ξRetroPlot<0]=-1
            ξRetroMask[ξRetroPlot>=0]=1
            ξProMask = np.zeros(len(ξProPlotarr))
            ξProMask[ξProPlot<0]=-1
            ξProMask[ξProPlot>=0]=1

            ξRetroGraphical = ξRetroPlotarr[:-1][np.diff(ξRetroMask)!=0]
            ξProGraphical = ξProPlotarr[:-1][np.diff(ξProMask)!=0]
            
            # Store number of roots estimated via graphical method for each m
            LenξRetroGraphical = len(ξRetroGraphical)
            LenξProGraphical = len(ξProGraphical)
            LenξRetro = len(self.ξRetro[mplot-1,:])
            LenξPro = len(self.ξPro[mplot-1,:])
            
            if LenξRetroGraphical < LenξRetro:
                raise RuntimeError('There may be double or spurious (retrograde) roots')
            if LenξProGraphical < LenξPro:
                raise RuntimeError('There may be double or spurious (prograde) roots')
            if LenξRetroGraphical > LenξRetro:
                raise RuntimeError('Some (retrograde) roots may be missing')
            if LenξProGraphical > LenξPro:
                raise RuntimeError('Some (prograde) roots may be missing')

            # Compare graphical estimates to fsolve root-finding results
            ξRetroGraphicalCheck = np.sum(np.abs(ξRetroGraphical - self.ξRetro[mplot-1,:]) > 10*gridres*ΔξRetroG)
            ξProGraphicalCheck = np.sum(np.abs(ξProGraphical - self.ξPro[mplot-1,:]) > 10*gridres*ΔξProG)

            if ξRetroGraphicalCheck != 0:
                raise RuntimeError('Some (retrograde) roots may be incorrect')
            if ξProGraphicalCheck != 0:
                raise RuntimeError('Some (prograde) roots may be incorrect')
            
        # Use the dispersion relation to compute the corresponding inviscid half-frequencies σ0

        self.σ0Retro = (1 + (self.ξRetro/(self.γ*np.pi))**2)**(-1/2)
        self.σ0Pro = -(1 + (self.ξPro/(self.γ*np.pi))**2)**(-1/2)

        self.ξ = np.stack((self.ξRetro,self.ξPro))
        self.σ0 = np.stack((self.σ0Retro,self.σ0Pro))
    
    def ξRetroFunc(ξ,m,γ):
        ξEqnp = ξ*sp.jv(-1 + m,ξ) + m*(-1 + np.sqrt(1 +
                ξ**2/(np.pi**2*γ**2)))*sp.jv(m,ξ)
        return ξEqnp

    def ξProFunc(ξ,m,γ):
        ξEqnp = ξ*sp.jv(-1 + m,ξ) + m*(-1 - np.sqrt(1 +
                ξ**2/(np.pi**2*γ**2)))*sp.jv(m,ξ)
        return ξEqnp
    
    def minimizeRayleigh(self, Ek, Pr, printVals=False):
        '''
        Minimizes modified Rayleigh number among the first M x K bulk modes

        Parameters
        ----------
        Ek : float
            Ekman number.
            Ratio of viscous to Coriolis forces on the container scale
            (Ek = ν/(Ω H^2))
        Pr : float
            Prandtl number.
            Ratio of viscosity to thermal diffusivity
            (Pr = ν/κ)
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

        References
        ----------
        Zhang, K., & Liao, X. (2009). The onset of convection in rotating circular cylinders 
            with experimental boundary conditions. Journal of Fluid Mechanics, 622, 63-73. 
            https://doi.org/10.1017/S002211200800517X

        Zhang, K., & Liao, X. (2017). Theory and Modeling of Rotating Fluids: Convection,
            Inertial Waves and Precession (Cambridge Monographs on Mechanics). Cambridge: 
            Cambridge University Press. https://doi.org/10.1017/9781139024853
        '''

        # Create grid of wavenumbers

        mArr = np.array([[m for k in range(1,self.K+1)] for m in range(1,self.M+1)])

        # Compute R1 for every mode m,k (1 <= m <= M, 1 <= k <= K).

        Sum = np.zeros((2,self.M,self.K))
        Sum2 = np.zeros((2,self.M,self.K))

        for j in range(1,self.J+1):
            Q = (np.pi**2*self.γ**2*self.β[...,j-1]**2)/(2*(-mArr**2 + self.β[...,j-1]**2)*(-self.β[...,j-1]**2+ self.ξ**2)**2*self.σ0**2)
            Sum += (Q*(np.pi**2 + self.β[...,j-1]**2/self.γ**2))/((4*Pr**2*self.σ0**2)/Ek**2 + (np.pi**2 + self.β[...,j-1]**2/self.γ**2)**2)
            Sum2 += Q/((np.pi**2 + self.β[...,j-1]**2/self.γ**2)**2 + (4*Pr**2*self.σ0**2)/Ek**2)

        with np.errstate(divide='ignore'):
            R1 = (np.sqrt(Ek)*((np.sqrt(Ek)*np.pi**2*(np.pi**2*self.γ**2 
                + mArr*(mArr - self.σ0)))/(self.σ0**2*np.sqrt(1 - self.σ0**2)) 
                - 2*mArr*self.σ0*(np.sqrt(1 - self.σ0) + np.sqrt(1 + self.σ0)) 
                + (mArr**2 + np.pi**2*self.γ**2)*((1 - self.σ0)**1.5 + (1 + self.σ0)**1.5 
                + (np.sqrt(1 - self.σ0**2)*np.sqrt(np.abs(self.σ0)))/self.γ)))/(4.*mArr**2*Sum*np.sqrt(1 - self.σ0**2))

        # Compute minimum R1 and corresponding inviscid half-frequency and wavenumbers.

        c0ind = np.unravel_index(np.argmin(R1, axis=None), R1.shape)
        σ0c = self.σ0[c0ind]
        ξ0c = self.ξ[c0ind]
        R1c = R1[c0ind]
        k0c = c0ind[2]+1
        m0c = c0ind[1]+1

        # Check that enough modes were searched
        if m0c > (self.M-1):
            raise ValueError('Not enough azimuthal modes were searched. Increase M and J.')
        if k0c > (self.K-1):
            raise ValueError('Not enough radial modes were searched. Increase K and J.')

        # Compute the corresponding viscously-corrected half-frequency

        σ1c = 0.5*(2*σ0c - 1/(np.pi**2*self.γ**2 + m0c*(m0c - σ0c))*(np.sqrt(1
                - σ0c**2)*np.sqrt(Ek))*(Sum2[c0ind]*(8*m0c**2*R1c*Pr*σ0c*np.sqrt(1 -
                σ0c**2))/Ek**1.5 + 2*σ0c*m0c*(np.sqrt(1 + σ0c) - np.sqrt(1 -
                σ0c)) + (m0c**2 + np.pi**2*self.γ**2)*((σ0c*np.sqrt(1 -
                σ0c**2))/(self.γ*np.sqrt(np.abs(σ0c))) - (1 + σ0c)**1.5 +
                (1 - σ0c)**1.5)))

        # Compute critical Rayleigh number, half-frequency, and azimuthal mode number for viscous (wall) convection

        Rwall = 2*np.sqrt((6*(9 + 5*np.sqrt(3)))/(5 + 3*np.sqrt(3)))*np.pi**2 + 73.8*Ek**(1/3)
        σwall = (-290.6*Ek**(4/3))/Pr + (np.sqrt(2 + np.sqrt(3))*(3 + np.sqrt(3))*Ek*np.pi**2)/((1 + np.sqrt(3))*Pr)
        mwall = self.γ*(np.sqrt(2 + np.sqrt(3))*np.pi - 27.76*Ek**(1/3))

        if printVals==True:
            
            print(f"Bulk: Rc = {R1c:.4f}, σc =  {σ1c:.4f}, mc = {m0c}, nc = 1, kc = {k0c}")
            print(f"Wall: Rc = {Rwall:.4f}, σc =  {σwall:.4f}, mc = {mwall:.4f}, nc = 1")
        
        else:
            return R1c, σ1c, m0c, k0c, Rwall, σwall, mwall
