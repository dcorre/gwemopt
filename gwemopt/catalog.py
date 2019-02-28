
import os, sys, copy
import h5py
import numpy as np
import healpy as hp

from scipy.stats import norm
from scipy.special import gammaincinv

from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = -1

from astropy.table import Table
from astropy.io import ascii
from astropy.cosmology import WMAP9 as cosmo
from astropy.table import Column
import astropy.units as u
import astropy.constants as c

def get_catalog(params, map_struct):

    if not os.path.isdir(params["catalogDir"]): os.makedirs(params["catalogDir"])
    catalogFile = os.path.join(params["catalogDir"],"%s.hdf5"%params["galaxy_catalog"])

    """AB Magnitude zero point."""
    MAB0 = -2.5 * np.log10(3631.e-23)
    pc_cm = 3.08568025e18
    const = 4. * np.pi * (10. * pc_cm)**2.

    if params["galaxy_catalog"] == "2MRS":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('J/ApJS/199/26/table3')

            ras, decs = cat["RAJ2000"], cat["DEJ2000"]
            cz = cat["cz"]
            magk = cat["Ktmag"]
            z = (u.Quantity(cat['cz']) / c.c).to(u.dimensionless_unscaled)

            completeness = 0.5
            alpha = -1.0
            MK_star = -23.55
            MK_max = MK_star + 2.5 * np.log10(gammaincinv(alpha + 2, completeness))
            MK = magk - cosmo.distmod(z)
            idx = (z > 0) & (MK < MK_max)
        
            ra, dec = ra[idx], dec[idx]
            z = z[idx]
            magk = magk[idx]            

            with h5py.File(catalogFile, 'w') as f:
                f.create_dataset('ra', data=ras)
                f.create_dataset('dec', data=decs)
                f.create_dataset('z', data=z)
                f.create_dataset('magk', data=magk)

        else:
            with h5py.File(catalogFile, 'r') as f:
                ra, dec = f['ra'][:], f['dec'][:]
                z = f['z'][:]
                magk = f['magk'][:]

        r = cosmo.luminosity_distance(z).to('Mpc').value
        mag = magk * 1.0
 
    elif params["galaxy_catalog"] == "GLADE":
        if not os.path.isfile(catalogFile):
            cat, = Vizier.get_catalogs('VII/281/glade2')

            ra, dec = cat["RAJ2000"], cat["DEJ2000"]
            distmpc, z = cat["Dist"], cat["z"]
            magb, magk = cat["Bmag"], cat["Kmag"]
            # Keep track of galaxy identifier
            GWGC, PGC, HyperLEDA = cat["GWGC"], cat["PGC"], cat["HyperLEDA"]
            _2MASS, SDSS = cat["_2MASS"], cat["SDSS-DR12"]

            idx = np.where(distmpc >= 0)[0]
            ra, dec = ra[idx], dec[idx]
            distmpc, z = distmpc[idx], z[idx]
            magb, magk = magb[idx], magk[idx]
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]

            with h5py.File(catalogFile, 'w') as f:
                f.create_dataset('ra', data=ra)
                f.create_dataset('dec', data=dec)
                f.create_dataset('distmpc', data=distmpc)
                f.create_dataset('magb', data=magb)
                f.create_dataset('magk', data=magk)
                f.create_dataset('z', data=z)
                # Add galaxy identifier
                f.create_dataset('GWGC', data=GWGC)
                f.create_dataset('PGC', data=PGC)
                f.create_dataset('HyperLEDA', data=HyperLEDA)
                f.create_dataset('2MASS', data=_2MASS)
                f.create_dataset('SDSS', data=SDSS)

        else:
            with h5py.File(catalogFile, 'r') as f:
                ra, dec = f['ra'][:], f['dec'][:]
                distmpc, z = f['distmpc'][:], f['z'][:]
                magb, magk = f['magb'][:], f['magk'][:]
                GWGC, PGC, HyperLEDA = f['GWGC'][:], f['PGC'][:], f['HyperLEDA'][:]
                _2MASS, SDSS = f['2MASS'][:], f['SDSS'][:]
                # Convert bytestring to unicode
                GWGC = GWGC.astype('U')
                PGC = PGC.astype('U')
                HyperLEDA = HyperLEDA.astype('U')
                _2MASS = _2MASS.astype('U')
                SDSS = SDSS.astype('U')

        if params["galaxy_grade"] != "3D":
            idx = np.where(~np.isnan(magb))[0]
            ra, dec, distmpc, magb, magk = ra[idx], dec[idx], distmpc[idx], magb[idx], magk[idx]

        r = distmpc * 1.0
        mag = magb * 1.0

    elif params["galaxy_catalog"] == "CLU":
        if not os.path.isfile(catalogFile):
            print("Please add %s."%catalogFile)
            exit(0)

        with h5py.File(catalogFile, 'r') as f:
            name = f['name'][:]
            ra, dec = f['ra'][:], f['dec'][:]
            sfr_fuv, mstar = f['sfr_fuv'][:], f['mstar'][:]
            distmpc, magb = f['distmpc'][:], f['magb'][:]
            a, b2a, pa = f['a'][:], f['b2a'][:], f['pa'][:]
            btc = f['btc'][:]

        idx = np.where(distmpc >= 0)[0]
        ra, dec = ra[idx], dec[idx]
        sfr_fuv, mstar = sfr_fuv[idx], mstar[idx]
        distmpc, magb = distmpc[idx], magb[idx]
        a, b2a, pa = a[idx], b2a[idx], pa[idx]
        btc = btc[idx]

        idx = np.where(~np.isnan(magb))[0]
        ra, dec = ra[idx], dec[idx]
        sfr_fuv, mstar = sfr_fuv[idx], mstar[idx]
        distmpc, magb = distmpc[idx], magb[idx]
        a, b2a, pa = a[idx], b2a[idx], pa[idx]
        btc = btc[idx]

        r = distmpc * 1.0
        mag = magb * 1.0

    L_nu =  const * 10.**((mag + MAB0)/(-2.5))
    L_nu = np.log10(L_nu)
    L_nu = L_nu**params["catalog_n"]
    Slum = L_nu / np.sum(L_nu)

    mlim, M_KNmin, M_KNmax = 22, -17, -12
    L_KNmin, L_KNmax = const * 10.**((M_KNmin + MAB0)/(-2.5)), const * 10.**((M_KNmax + MAB0)/(-2.5))

    Llim = 4. * np.pi * (r * 1e6 * pc_cm)**2. * 10.**((mlim + MAB0)/(-2.5))
    Sdet = (L_KNmax-Llim)/(L_KNmax-L_KNmin)
    Sdet[Sdet<0.01] = 0.01
    Sdet[Sdet>1.0] = 1.0

    theta = 0.5 * np.pi - dec * 2 * np.pi /360.0
    phi = ra * 2 * np.pi /360.0
    ipix = hp.ang2pix(map_struct["nside"], ra, dec, lonlat=True)
    
    if "distnorm" in map_struct:
        """
        r_test = np.linspace(0,5000,5000)
        print ("Sum Prob 2D: %e   %e" % (np.sum(map_struct["prob"][ipix]),np.sum(map_struct["prob"])) )
        for i in range(len(ipix)):
            print ("Prob 2D: %e  Norm: %e   Prob dist: %e   Sum PDF dist: %e" % (map_struct["prob"][ipix][i], map_struct["distnorm"][ipix][i], map_struct["distnorm"][ipix][i] * norm(map_struct["distmu"][ipix][i], map_struct["distsigma"][ipix][i]).pdf(r[i]), np.sum(map_struct["distnorm"][ipix][i] * norm(map_struct["distmu"][ipix][i], map_struct["distsigma"][ipix][i]).pdf(r_test))))
        """
        Sloc = map_struct["prob"][ipix] * (map_struct["distnorm"][ipix] * norm(map_struct["distmu"][ipix], map_struct["distsigma"][ipix]).pdf(r))**params["powerlaw_dist_exp"] / map_struct["pixarea"]
        #Sloc = copy.copy(map_struct["prob"][ipix])
        #print (len(Sloc), len(Sloc[np.isnan(Sloc)]))
        Sloc[np.isnan(Sloc)] = 0
    else:
        Sloc = copy.copy(map_struct["prob"][ipix])
    
    S = Sloc*Slum*Sdet
    prob = np.zeros(map_struct["prob"].shape)
    if params["galaxy_grade"] == "loc":
        prob[ipix] = prob[ipix] + Sloc
    else:
        prob[ipix] = prob[ipix] + S
    prob = prob / np.sum(prob)

    map_struct['prob_catalog'] = prob
    if params["doUseCatalog"]:
        map_struct['prob'] = prob 

    if params["galaxy_grade"] == "loc":
        idx = np.where(~np.isnan(Sloc))[0]
        ra, dec, Sloc, distmpc, z = ra[idx], dec[idx], Sloc[idx], distmpc[idx], z[idx]
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]

        Sthresh = np.max(Sloc)*0.01
        idx = np.where(Sloc >= Sthresh)[0]
        ra, dec, Sloc, distmpc, z = ra[idx], dec[idx], Sloc[idx], distmpc[idx], z[idx]
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]

        idx = np.argsort(Sloc)[::-1]
        ra, dec, Sloc, distmpc, z = ra[idx], dec[idx], Sloc[idx], distmpc[idx], z[idx]
        GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
        _2MASS, SDSS = _2MASS[idx], SDSS[idx]

    else:
        
        idx = np.where(~np.isnan(S))[0]
        ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]

        Sthresh = np.max(S)*0.01
        idx = np.where(S >= Sthresh)[0]
        ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx] 

        idx = np.argsort(S)[::-1]
        ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx] 
        
    if len(ra) > 50:
        print('Cutting catalog to top 50 galaxies...')
        idx = np.arange(50).astype(int)
        if params["galaxy_grade"] == "3D":
            ra, dec, Sloc, distmpc, z = ra[idx], dec[idx], Sloc[idx], distmpc[idx], z[idx]
            GWGC, PGC, HyperLEDA = GWGC[idx], PGC[idx], HyperLEDA[idx]
            _2MASS, SDSS = _2MASS[idx], SDSS[idx]
        else:
            ra, dec, Sloc, S = ra[idx], dec[idx], Sloc[idx], S[idx]

    catalogfile = os.path.join(params["outputDir"],'catalog.csv')
    fid = open(catalogfile,'w')
    cnt = 1
    if params["galaxy_grade"] == "loc":  
        for a, b, c, d, e, f, g, h, i, j in zip(ra, dec, Sloc, distmpc, z, GWGC, PGC, HyperLEDA, _2MASS, SDSS):
            fid.write("%d, %.5f, %.5f, %.5e, %.2f, %.4f, %s, %s, %s, %s, %s\n"%(cnt,a,b,c,d,e,f,g,h,i,j))
            cnt = cnt + 1
    else:
        for a, b, c, d in zip(ra, dec, Sloc, S):
            fid.write("%d %.5f %.5f %.5e %.5e\n"%(cnt,b,c,d))
            cnt = cnt + 1

    fid.close()

    return map_struct

