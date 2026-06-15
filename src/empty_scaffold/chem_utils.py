from __future__ import annotations

from typing import Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

NO_SCAFFOLD_LABEL = "No_Scaffold"
INVALID_SCAFFOLD_LABELS = {"Invalid_SMILES", "Error"}


def standardize_mol(mol: Chem.Mol) -> Chem.Mol:
    """Standardize an RDKit molecule for full-analysis workflows."""
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:
        pass
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return "Invalid_SMILES"
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return NO_SCAFFOLD_LABEL
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return "Invalid_SMILES"


def is_no_scaffold(scaffold: str) -> bool:
    return scaffold == NO_SCAFFOLD_LABEL


def is_invalid_scaffold(scaffold: str) -> bool:
    return scaffold in INVALID_SCAFFOLD_LABELS


def morgan_fp_array(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
        return np.asarray(fp, dtype=np.float32)
    except Exception:
        return None


def rdkit_descriptor_array(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return np.asarray(rdMolDescriptors.Properties().ComputeProperties(mol), dtype=np.float32)
    except Exception:
        return None


def morgan_fp_bitvect(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True)
    except Exception:
        return None


def max_tanimoto_to_train(query_smiles, train_smiles, radius: int = 2, n_bits: int = 2048):
    train_fps = [morgan_fp_bitvect(s, radius, n_bits) for s in train_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]
    if not train_fps:
        return []
    values = []
    for smi in query_smiles:
        fp = morgan_fp_bitvect(smi, radius, n_bits)
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        if sims:
            values.append(float(max(sims)))
    return values
