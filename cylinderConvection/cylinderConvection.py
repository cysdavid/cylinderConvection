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

    def __init__(self, Γ, M, K, J, debug=False):
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
        debug : bool, optional
            Sets option for errors.
            If true, errors will be suppresssed.

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
        self.debug = debug

        if M > 90:
            self.raiseError(ValueError,'M cannot be greater than 90.')
        if K > 90:
            self.raiseError(ValueError,'K cannot be greater than 90.')

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
                self.raiseError(ValueError,msgRetro)
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
                self.raiseError(ValueError,msgPro)
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
                self.raiseError(RuntimeError,'There may be double or spurious (retrograde) roots')
            if LenξProGraphical < LenξPro:
                self.raiseError(RuntimeError,'There may be double or spurious (prograde) roots')
            if LenξRetroGraphical > LenξRetro:
                self.raiseError(RuntimeError,'Some (retrograde) roots may be missing')
            if LenξProGraphical > LenξPro:
                self.raiseError(RuntimeError,'Some (prograde) roots may be missing')

            # Compare graphical estimates to fsolve root-finding results
            ξRetroGraphicalCheck = np.sum(np.abs(ξRetroGraphical - self.ξRetro[mplot-1,:]) > 10*gridres*ΔξRetroG)
            ξProGraphicalCheck = np.sum(np.abs(ξProGraphical - self.ξPro[mplot-1,:]) > 10*gridres*ΔξProG)

            if ξRetroGraphicalCheck != 0:
                self.raiseError(RuntimeError,'Some (retrograde) roots may be incorrect')
            if ξProGraphicalCheck != 0:
                self.raiseError(RuntimeError,'Some (prograde) roots may be incorrect')
            
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
    
    def raiseError(self,errorClass, msg):
        if not self.debug:
            raise errorClass(msg)
        else:
            print(msg)
    
    def minimizeRayleigh(self, E, Pr, printVals=False): 
        '''
        Minimizes modified Rayleigh number among the first M x K bulk modes

        Parameters
        ----------
        E : float
            Ekman number.
            Ratio of viscous to Coriolis forces on the container scale
            (E = ν/(2 Ω H^2))
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
        Rac : float
            Minimum Rayleigh number among the first M x K bulk modes.
            (Ra = α ΔT g H^3/(ν κ))
        R1c : float
            Minimum MODIFIED Rayleigh number among the first M x K bulk modes.
            Defined as the Rayleigh number scaled by half the inverse Ekman number.
            (R = 2 Ra E = α ΔT g H/(Ω κ))
        σ1c : float
            Half-frequency of the bulk mode with R = R1c,
            scaled by the rotational rate Ω.
        m0c : float
            Azimuthal wavenumber of the bulk mode with R = R1c.
        k0c : float
            Radial wavenumber of the bulk mode with R = R1c.
        RaWall : float
            Minimum Rayleigh number among sidewall-localized (viscous) convective modes. 
            Obtained by rescaling the asymptotic law in a channel.
        RWall : float
            Minimum MODIFIED Rayleigh number among sidewall-localized (viscous) convective modes. 
            Obtained by rescaling the asymptotic law in a channel.
            (RWall = 2 RaWall E)
        σWall : float
            Half-frequency of the wall mode with R = RWall,
            scaled by the rotational rate Ω.
            Obtained by rescaling the asymptotic law in a channel.
        mWall : float
            Azimuthal wavenumber of the wall mode with R = RWall
            Obtained by rescaling the asymptotic law in a channel.

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

        self.Pr = Pr
        self.E = E

        # Let self.Ek = 2 E = ν/(Ω H^2)
        
        self.Ek = 2*E

        # Create grid of wavenumbers

        mArr = np.array([[m for k in range(1,self.K+1)] for m in range(1,self.M+1)])

        # Compute R1 for every mode m,k (1 <= m <= M, 1 <= k <= K).

        Sum = np.zeros((2,self.M,self.K))
        Sum2 = np.zeros((2,self.M,self.K))

        for j in range(1,self.J+1):
            Q = (np.pi**2*self.γ**2*self.β[...,j-1]**2)/(2*(-mArr**2 + self.β[...,j-1]**2)*(-self.β[...,j-1]**2+ self.ξ**2)**2*self.σ0**2)
            Sum += (Q*(np.pi**2 + self.β[...,j-1]**2/self.γ**2))/((4*self.Pr**2*self.σ0**2)/self.Ek**2 + (np.pi**2 + self.β[...,j-1]**2/self.γ**2)**2)
            Sum2 += Q/((np.pi**2 + self.β[...,j-1]**2/self.γ**2)**2 + (4*self.Pr**2*self.σ0**2)/self.Ek**2)

        with np.errstate(divide='ignore'):
            R1 = (np.sqrt(self.Ek)*((np.sqrt(self.Ek)*np.pi**2*(np.pi**2*self.γ**2 
                + mArr*(mArr - self.σ0)))/(self.σ0**2*np.sqrt(1 - self.σ0**2)) 
                - 2*mArr*self.σ0*(np.sqrt(1 - self.σ0) + np.sqrt(1 + self.σ0)) 
                + (mArr**2 + np.pi**2*self.γ**2)*((1 - self.σ0)**1.5 + (1 + self.σ0)**1.5 
                + (np.sqrt(1 - self.σ0**2)*np.sqrt(np.abs(self.σ0)))/self.γ)))/(4.*mArr**2*Sum*np.sqrt(1 - self.σ0**2))

        # Compute minimum R1 and corresponding inviscid half-frequency and wavenumbers.

        c0ind = np.unravel_index(np.argmin(R1, axis=None), R1.shape)
        self.σ0c = self.σ0[c0ind]
        self.ξ0c = self.ξ[c0ind]
        self.R1c = R1[c0ind]
        self.k0c = c0ind[2]+1
        self.m0c = c0ind[1]+1

        # Check that enough modes were searched
        if self.m0c > (self.M-1):
            if self.J < self.M:
                self.raiseError(ValueError,'Not enough temperature modes were included. Increase J. If no improvement, increase M as well.')
            else:
                self.raiseError(ValueError,'Not enough azimuthal wavenumbers were searched. Increase M and J.')
        if self.k0c > (self.K-1):
            if self.J < self.K:
                self.raiseError(ValueError,'Not enough temperature modes were included. Increase J. If no improvement, increase K as well.')
            else:
                self.raiseError(ValueError,'Not enough radial wavenumbers were searched. Increase K and J.')

        # Compute the corresponding viscously-corrected half-frequency

        self.σ1c = 0.5*(2*self.σ0c - 1/(np.pi**2*self.γ**2 + self.m0c*(self.m0c - self.σ0c))*(np.sqrt(1
                - self.σ0c**2)*np.sqrt(self.Ek))*(Sum2[c0ind]*(8*self.m0c**2*self.R1c*self.Pr*self.σ0c*np.sqrt(1 -
                self.σ0c**2))/self.Ek**1.5 + 2*self.σ0c*self.m0c*(np.sqrt(1 + self.σ0c) - np.sqrt(1 -
                self.σ0c)) + (self.m0c**2 + np.pi**2*self.γ**2)*((self.σ0c*np.sqrt(1 -
                self.σ0c**2))/(self.γ*np.sqrt(np.abs(self.σ0c))) - (1 + self.σ0c)**1.5 +
                (1 - self.σ0c)**1.5)))

        # Compute modified critical Rayleigh number, half-frequency, 
        # and azimuthal mode number for viscous (wall) convection

        self.RWall = 2*np.pi**2*np.sqrt(6*np.sqrt(3)) + 73.80*self.Ek**(1/3)
        self.σWall = (-290.6*self.Ek**(4/3))/self.Pr + (np.sqrt(2 + np.sqrt(3))*(3 + np.sqrt(3))*self.Ek*np.pi**2)/((1 + np.sqrt(3))*self.Pr)
        self.mWall = self.γ*(np.sqrt(2 + np.sqrt(3))*np.pi - 27.76*self.Ek**(1/3))

        # Compute self.Rac = self.R1c/self.Ek = self.R1c/(2 E)
        self.Rac = self.R1c/self.Ek
        self.RaWall = self.RWall/self.Ek

        if printVals==True:
            
            print(f"Bulk: self.Rac = {self.Rac:.4e}, σc =  {self.σ1c:.4f}, mc = {self.m0c}, nc = 1, kc = {self.k0c}")
            print(f"Wall: self.Rac = {self.RaWall:.4e}, σc =  {self.σWall:.4f}, mc = {self.mWall:.4f}, nc = 1")
        
        else:
            return self.Rac, self.R1c, self.σ1c, self.m0c, self.k0c, self.RaWall, self.RWall, self.σWall, self.mWall

    def grid(self,ns,nφ,nz,t):
        '''
        Convenience function to produce 4D meshgrids of 
        points in cylindrical radius (s), azimuth(φ),
        height (z), and time (t) over the entire cylinder
        at specified time(s).

        Parameters
        ----------
        ns : int
            Number of radial grid points on interval [0,Γ/2].
        nφ : int
            Number of azimuthal grid points on interval [0,2π].
        nz : int
            Number of vertical grid points on interval [0,1].
        t : float, array
            Time(s) scaled by inverse rotation rate 1/Ω.

        Returns
        -------
        grids : dict
            Dictionary of 4D arrays of points in cylindrical 
            radius, azimuth, height, and time corresponding to the keys
            's', 'phi', 'z', and 't', respectively, in addition to the
            (converted) Cartesian positions corresponding to the keys 
            'x' and 'y'.

        Notes
        -----
        For each array of grid points, the axes 0, 1, 2, 3 correspond to
        s, φ, z, t, respectively. E.g., grids['s'][-1,0,0,0] = Γ/2,
        grids['φ'][-1,0,0,0] = 2 π,  grids['z'][0,0,-1,0] = 1. 
        
        '''

        sarr1d = np.linspace(0,self.γ,ns)
        sarr1d[0] = 1e-4*self.γ
        φarr1d = np.linspace(0,2*np.pi,nφ)
        zarr1d = np.linspace(0,1,nz)

        sGrid = np.tensordot(np.tensordot(np.tensordot(sarr1d, np.ones(nφ), 0), np.ones(nz), 0), np.ones(np.size(t)), 0)
        φGrid = np.tensordot(np.tensordot(np.tensordot(np.ones(ns), φarr1d, 0), np.ones(nz), 0), np.ones(np.size(t)), 0)
        zGrid = np.tensordot(np.tensordot(np.tensordot(np.ones(ns), np.ones(nφ), 0), zarr1d, 0), np.ones(np.size(t)), 0)
        tGrid = np.tensordot(np.tensordot(np.tensordot(np.ones(ns), np.ones(nφ), 0), np.ones(nz), 0), t*np.ones(np.size(t)), 0)

        xGrid = sGrid * np.cos(φGrid)
        yGrid = sGrid * np.sin(φGrid)

        grids = {
                's':sGrid, 'phi':φGrid, 'z':zGrid,
                'x':xGrid, 'y':yGrid, 't':tGrid
                }

        return grids
    
    def gridData(self,ns,nφ,nz,t):
        '''
        Evaluates the asymptotic solution for the velocity
        field of the onset oscillatory mode at grid points
        of a cylindrical mesh at specified time(s).

        Parameters
        ----------
        ns : int
            Number of radial grid points on interval [0,Γ/2].
        nφ : int
            Number of azimuthal grid points on interval [0,2π].
        nz : int
            Number of vertical grid points on interval [0,1].
        t : float, array
            Time(s) scaled by inverse rotation rate 1/Ω.

        Returns
        -------
        grids : dict
            Dictionary of 4D arrays of points in cylindrical 
            radius, azimuth, height, and time corresponding to the keys
            's', 'phi', 'z', and 't', respectively; (converted) Cartesian
            positions corresponding to the keys 'x' and 'y'; and velocity
            grid data in cylindrical coordinates, corresponding to the keys
            'us', 'uphi', and 'uz', respectively.  

        Notes
        -----
        For each array of grid points, the axes 0, 1, 2, 3 correspond to
        s, φ, z, t, respectively. E.g., grids['s'][-1,0,0,0] = Γ/2,
        grids['φ'][-1,0,0,0] = 2 π,  grids['z'][0,0,-1,0] = 1. 
        '''
        
        grids = self.grid(ns,nφ,nz,t)
        sGrid,φGrid,zGrid,xGrid,yGrid,tGrid = grids.values()

        usGrid = self.us(sGrid,φGrid,zGrid,tGrid)
        uφGrid = self.uφ(sGrid,φGrid,zGrid,tGrid)
        uzGrid = self.uz(sGrid,φGrid,zGrid,tGrid)

        gdata = {
                's':sGrid, 'phi':φGrid, 'z':zGrid,
                'x':xGrid, 'y':yGrid, 't':tGrid,
                'us':usGrid, 'uphi':uφGrid, 'uz':uzGrid
                }
        
        return gdata
    
    def synthDoppler(self,s1,φ1,z1,s2,φ2,z2,t,gates):
        '''
        Generates synthetic ultrasonic Doppler velocimetry
        (UDV) data using the asymptotic solution for the
        critical oscillatory mode. 

        Parameters
        ----------
        s1 : float
            Radial position of the UDV transducer
            scaled by the tank height.
        φ1 : float
            Azimuthal position of the UDV transducer
            on interval [0,2π].
        z1 : float
            Vertical position of the UDV transducer
            scaled by the tank height.
        s2 : float
            Radial position of the ultrasonic beam
            path endpoint scaled by the tank height.
        φ2 : float
            Azimuthal position of the ultrasonic beam
            path endpoint on interval [0,2π].
        z2 : float
            Vertical position of the ultrasonic beam
            path endpoint scaled by the tank height.
        t : float, array
            Time(s) scaled by inverse rotation rate 1/Ω.
        gates : int
            Number of gates of the ultrasonic beam
            (spatial resolution).

        Returns
        -------
        gdataDop : dict
            Dictionary of 2D arrays including meshgrids
            of times, distances along the ultrasonic beam path,
            and the component of velocity along the beam path
            (with positive values for flow towards the transducer),
            corresponding to the keys 't', 'l', and 'uDop', respectively.
        beamPath : dict
            Dictionary of 1D arrays of the Cartesian (x,y,z) coordinates
            of all sample volumes in the ultrasonic beam path, corresponding
            to the keys 'x', 'y', 'z', respectively.

        Notes
        -----
        For each array of grid points, the axes 0 and 1 correspond to
        time t and distance l along the beam path respectively.

        '''

        q = np.linspace(0,1,gates)
        qGrid = np.tensordot(np.ones(np.size(t)), q, 0)
        tGrid = np.tensordot(t*np.ones(np.size(t)), np.ones(gates), 0)

        xBGrid = s1*np.cos(φ1) + qGrid*(-(s1*np.cos(φ1)) + s2*np.cos(φ2))
        yBGrid = s1*np.sin(φ1) + qGrid*(-(s1*np.sin(φ1)) + s2*np.sin(φ2))                                                             
        zBGrid = z1 + qGrid*(-z1 + z2)

        L = np.sqrt((xBGrid[0,0]-xBGrid[0,-1])**2 + (yBGrid[0,0]-yBGrid[0,-1])**2 + (zBGrid[0,0]-zBGrid[0,-1])**2)
        lGrid = L*qGrid

        beamX = (-(s1*np.cos(φ1)) + s2*np.cos(φ2))/np.sqrt(s1**2 + s2**2 + z1**2 - 2*z1*z2 + z2**2 - 2*s1*s2*np.cos(φ1 - φ2))
        beamY = (-(s1*np.sin(φ1)) + s2*np.sin(φ2))/np.sqrt(s1**2 + s2**2 + z1**2 - 2*z1*z2 + z2**2 - 2*s1*s2*np.cos(φ1 - φ2)) 
        beamZ = (-z1 + z2)/np.sqrt(s1**2 + s2**2 + z1**2 - 2*z1*z2 + z2**2 - 2*s1*s2*np.cos(φ1 - φ2))
                                                                                                                                                                                                                                                                                                                                      
        uxBeamGrid = (xBGrid*self.us(np.sqrt(xBGrid**2 + yBGrid**2),np.arctan2(yBGrid,xBGrid),zBGrid,tGrid) 
                  - yBGrid*self.uφ(np.sqrt(xBGrid**2 + yBGrid**2),np.arctan2(yBGrid,xBGrid),zBGrid,tGrid))/np.sqrt(xBGrid**2 + yBGrid**2)
        uyBeamGrid = (yBGrid*self.us(np.sqrt(xBGrid**2 + yBGrid**2),np.arctan2(yBGrid,xBGrid),zBGrid,tGrid)
                  + xBGrid*self.uφ(np.sqrt(xBGrid**2 + yBGrid**2),np.arctan2(yBGrid,xBGrid),zBGrid,tGrid))/np.sqrt(xBGrid**2 + yBGrid**2)
        uzBeamGrid = self.uz(np.sqrt(xBGrid**2 + yBGrid**2),np.arctan2(yBGrid,xBGrid),zBGrid,tGrid)

        uDopGrid = -(beamX*uxBeamGrid + beamY*uyBeamGrid + beamZ*uzBeamGrid)

        gdataDop = {
                   't':tGrid, 'l':lGrid, 'uDop':uDopGrid
                   }
        
        beamPath = {
                    'x':xBGrid[0,:], 'y':yBGrid[0,:], 'z':zBGrid[0,:]
                    }
        
        return gdataDop, beamPath

    def χ(self):
        χ = np.sqrt(np.abs(self.σ0c)/self.Ek)*(1 + 1j*self.σ0c/np.abs(self.σ0c))
        return χ

    def χp(self):
        χp = (1 + 1j)*np.sqrt((1 + self.σ0c)/self.Ek)
        return χp

    def χm(self):
        χm = (1 - 1j)*np.sqrt((1 - self.σ0c)/self.Ek)
        return χm
                                
    def us(self,s,φ,z,t):
        stilde = self.ξ0c * s/self.γ
        us = np.real(1j/(4.*(1 - self.σ0c**2))*(-2*((self.σ0c*self.ξ0c)/self.γ*sp.jv(self.m0c - 1,stilde) 
            + (self.m0c*(1 - self.σ0c))/s*sp.jv(self.m0c,stilde))*np.cos(np.pi*z)
            + ((self.ξ0c*(self.σ0c - 1))/self.γ*sp.jv(self.m0c - 1,stilde)
            + (2*self.m0c*(1 - self.σ0c))/s*sp.jv(self.m0c,stilde))*(np.exp(-self.χp()*z) 
            - np.exp(-self.χp()*(1 - z)))
            + (self.ξ0c*(self.σ0c + 1))/self.γ*sp.jv(self.m0c - 1,stilde)*(np.exp(-self.χm()*z)
            - np.exp(-self.χm()*(1 - z))))*np.exp(1j*(2*self.σ1c*t + self.m0c*φ)))
        return us

    def uφ(self,s,φ,z,t):
        stilde = self.ξ0c * s/self.γ
        uφ = np.real(1/(4.*(1 - self.σ0c**2))*(2*(self.ξ0c/self.γ*sp.jv(self.m0c - 1,stilde)
            - (self.m0c*(1 - self.σ0c))/s*sp.jv(self.m0c,stilde))*np.cos(np.pi*z)
            + (self.m0c*sp.jv(self.m0c,stilde))/(2.*self.σ0c*self.γ)*np.cos(np.pi*z)*np.exp(-self.χ()*(self.γ - s))
            + ((self.ξ0c*(self.σ0c - 1))/self.γ*sp.jv(self.m0c - 1,stilde)
            + (2*self.m0c*(1 - self.σ0c))/s*sp.jv(self.m0c,stilde))*(np.exp(-self.χp()*z)
            - np.exp(-self.χp()*(1 - z)))
            - (self.ξ0c*(self.σ0c + 1))/self.γ*sp.jv(self.m0c - 1,stilde)*(np.exp(-self.χm()*z)
            - np.exp(-self.χm()*(1 - z))))*np.exp(1j*(2*self.σ1c*t + self.m0c*φ)))
        return uφ

    def uz(self,s,φ,z,t):
        stilde = self.ξ0c * s/self.γ
        uz = np.real(-(1j/(2.*self.σ0c))*(sp.jv(self.m0c,stilde)
            - sp.jv(self.m0c,self.ξ0c)*np.exp(-self.χ()*(self.γ - s)))*np.sin(np.pi*z)*np.exp(1j*(2*self.σ1c*t + self.m0c*φ)))
        return uz