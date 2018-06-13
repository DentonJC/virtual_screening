from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
DrawingOptions.atomLabelFontSize = 50
DrawingOptions.dotsPerAngstrom = 100
DrawingOptions.bondLineWidth = 3


def plot_mol(smiles, name = "mol.png"):
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(mol, name)

if __name__=='__main__':
    smiles = "C#C[NH2+][C@@H]1CCc2c1cccc2"
    plot_mol(smiles)
