import os
import tempfile
import subprocess
import numpy as np

def molprint2d_fingerprint(mpd, d=50):
    """
    mpd: str molprint2D repr of mol
    D - size of fingerprint 
    """
    v = [0] * d
    for substructure in mpd.split('\t')[1:]:
        if len(substructure) < 4:
            continue
        v[hash(substructure)%d] += 1
    return np.array(v)


def transform(smiles):
    """
    smiles: str
    molecule smiles
    """
    temp_dir = tempfile.mkdtemp()
    smiles_filename = os.path.join(temp_dir, "smiles.smi")
    output_filename = os.path.join(temp_dir, "output.mpd")

    with open(smiles_filename, 'w') as f_smi:
        f_smi.write(smiles)
    
    subprocess.call(["babel", smiles_filename, output_filename])
    
    with open(output_filename, 'r') as f_mpd:
        mpd = f_mpd.read()
        
    os.remove(smiles_filename)
    os.remove(output_filename)
    
    return mpd

        
if __name__=='__main__':
    kwasy = ["N(=O)O", "[N+](=O)(O)[O-]"]
    for kwas in kwasy:
        print(transform(kwas))
