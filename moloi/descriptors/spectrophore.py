import numpy as np
from openbabel import OBSpectrophore
import moloi.descriptors.mpd
from pybel import Molecule


def molprint2D(mol, d=50):
    """
    Calculates molprint2D representation.
    
    mol: pybel.Molecule
    D: int - size of fingerprint, Staszek says should be between 50 and 500
    
    returns: str
    molprint2D representation of mol
    """
    
    result = mpd.molprint2d_fingerprint(mpd.transform(mol.__str__()), d)
    return result
 

def spectro_average(samples):
    return np.mean(np.array(samples), axis=0)

        
def smiles_to_spectrophore(mol, n_samples=10, combining_method=None):
    """
    Calculates spectrophore representation. Explicilty adds all hydrogens to mol. If mol is not 3D makes it 3D multiple times
    (calculating 3D conformation is not deterministic) and calculates spectrophore for each conformation, then combines
    all calculated representations using the combining method provided. If mol is 3D everything is deterministic and spectrophore
    repr does not need to be combined in any way.
    Note: the mol object is modified during the run of the function.
    
    mol: pybel.Molecule
    
    n_sampels: int
    if given molecule does not have 3D structure calculated it will be which is not deterministic.
    How many samples should be processed?
    
    combining_method: function, gets list of spectrophores (std:vectors of doubles), outputs combined representation as np.array
    
    
    returns: np.array
    spectrophore representation of mol
    """
    # print "parsing", mol
    #mol.addh()
    spectrophore_calculator = OBSpectrophore()
    if mol.dim == 3:
        spectro = np.array(spectrophore_calculator.GetSpectrophore(mol.OBMol))
    else:
        assert n_samples >= 1, "The mol is not 3D, a positive number of samples needs to be defined"
        spectro = []
        for i in range(n_samples):
            mol.make3D()
            spectro.append(np.array(spectrophore_calculator.GetSpectrophore(mol.OBMol)))
        if combining_method:
            spectro = combining_method(spectro)
    return spectro


def random(mol, shape=10):
    """
    Return random representation.
    use _set_seed(seed) to have repeatitive datasets
    
    mol: pybel.Molecule
    
    returns: np.array
    random representation of mol
    """        
    return np.random.normal(size=shape)


def _set_seed(seed=666):
    np.random.seed(seed)

"""
if __name__ == "__main__":
    #features = smiles_to_mordred(pd.Series({"lol": "CC", "lol2": "CCN"}))
    features = smiles_to_spectrophore(Molecule("lol2"), n_samples=10, combining_method=None)
    print(features)
"""
