from typing import Dict

import os
import os.path as osp

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdPartialCharges

import pandas as pd
import numpy as np
import h5py
from tqdm.auto import tqdm


allowable_features = {
    'possible_atomic_num_list' : list(range(1, 35)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
    'possible_is_rotable_list': [False, True],
    'possible_ring_size_list': [3, 4, 5, 6, 7, 8, 'misc'],
}


_CARBON_EN = 2.55
_HBOND_DONOR_ATOMIC_NUMBERS = {7, 8, 16}
# Strict rotatable bond SMARTS (Oprea definition).
# Acyclic single bond where neither end is terminal, triple-bonded, methyl,
# trihalomethyl (CF3/CCl3/CBr3), or t-butyl.  Amide-like C(=X)–Y bonds
# (X,Y in {N,O,S}) are excluded via one-sided constraints that SMARTS
# matching applies symmetrically over both atom orderings.
_ROT_BASE = (                       # filters applied to BOTH end-atoms
    '!$(*#*)'                       # not in a triple bond
    '&!D1'                          # not terminal
    '&!$(C(F)(F)F)'                 # not CF3
    '&!$(C(Cl)(Cl)Cl)'             # not CCl3
    '&!$(C(Br)(Br)Br)'             # not CBr3
    '&!$(C([CH3])([CH3])[CH3])'    # not t-butyl centre
    '&!$([CH3])'                    # not methyl
)
_ROT_AMIDE = (                      # extra filters on ONE end-atom
    '&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])'   # amide-like C→heteroatom
    '&!$([#7,O,S!D1]-!@[CD3]=[N,O,S])'     # heteroatom→amide-like C
    '&!$([CD3](=[N+])-!@[#7!D1])'           # guanidinium-like C→N
    '&!$([#7!D1]-!@[CD3]=[N+])'             # N→guanidinium-like C
)
_ROTATABLE_BOND_SMARTS = Chem.MolFromSmarts(
    f'[{_ROT_BASE}{_ROT_AMIDE}]'
    '-,:;!@'                        # acyclic single / aromatic bond
    f'[{_ROT_BASE}]'
)
RWPE_DIM = 12
EN_DIM = 1
GC_DIM = 1
COORD_DIM = 3
NODE_CONTINUOUS_DIM = RWPE_DIM + EN_DIM + GC_DIM + COORD_DIM
# Pauling electronegativity by atomic number; unlisted elements default to _CARBON_EN.
_PAULING_EN = {
    1: 2.20,    # H
    3: 0.98,    # Li
    4: 1.57,    # Be
    5: 2.04,    # B
    6: 2.55,    # C
    7: 3.04,    # N
    8: 3.44,    # O
    9: 3.98,    # F
    11: 0.93,   # Na
    12: 1.31,   # Mg
    13: 1.61,   # Al
    14: 1.90,   # Si
    15: 2.19,   # P
    16: 2.58,   # S
    17: 3.16,   # Cl
    19: 0.82,   # K
    20: 1.00,   # Ca
    21: 1.36,   # Sc
    22: 1.54,   # Ti
    23: 1.63,   # V
    24: 1.66,   # Cr
    25: 1.55,   # Mn
    26: 1.83,   # Fe
    27: 1.88,   # Co
    28: 1.91,   # Ni
    29: 1.90,   # Cu
    30: 1.65,   # Zn
    31: 1.81,   # Ga
    32: 2.01,   # Ge
    33: 2.18,   # As
    34: 2.55,   # Se
    35: 2.96,   # Br
}


def _is_active_hydrogen(atom):
    if atom.GetAtomicNum() != 1:
        return False
    neighbors = atom.GetNeighbors()
    return len(neighbors) == 1 and neighbors[0].GetAtomicNum() in _HBOND_DONOR_ATOMIC_NUMBERS


def _non_active_hydrogen_count(atom):
    non_active_hydrogen_count = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 1:
            continue
        if not _is_active_hydrogen(neighbor):
            non_active_hydrogen_count += 1
    return non_active_hydrogen_count


def _active_hydrogen_mask(mol_with_hs):
    keep_atom = np.zeros(mol_with_hs.GetNumAtoms(), dtype=np.bool_)

    for atom in mol_with_hs.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() != 1:
            keep_atom[idx] = True
            continue

        neighbors = atom.GetNeighbors()
        if len(neighbors) != 1 or not _is_active_hydrogen(atom):
            continue
        parent = neighbors[0]
        if parent.GetAtomicNum() in _HBOND_DONOR_ATOMIC_NUMBERS:
            keep_atom[idx] = True

    return keep_atom


def _get_centered_en(atom):
    en = _PAULING_EN.get(atom.GetAtomicNum())
    if en is None:
        en = _CARBON_EN
    return float(en - _CARBON_EN)


def _get_gasteiger_charge(atom) -> float:
    """Return the Gasteiger partial charge for an atom, falling back to 0.0 on failure."""
    try:
        gc = atom.GetDoubleProp('_GasteigerCharge')
        if gc != gc or abs(gc) > 4.0:  # NaN check and outlier guard
            return 0.0
        return float(gc)
    except KeyError:
        return 0.0


def _compute_rwpe(num_nodes: int, edge_index: np.ndarray, rwpe_dim: int = RWPE_DIM) -> np.ndarray:
    """Compute random-walk positional encoding using return probabilities."""
    if num_nodes <= 0:
        return np.zeros((0, rwpe_dim), dtype=np.float16)

    adjacency = np.eye(num_nodes, dtype=np.float32)
    if edge_index.size > 0:
        src = edge_index[0]
        dst = edge_index[1]
        adjacency[src, dst] = 1.0

    degree = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    nonzero_degree = degree > 0
    transition[nonzero_degree] = adjacency[nonzero_degree] / degree[nonzero_degree, None]

    rwpe = np.zeros((num_nodes, rwpe_dim), dtype=np.float32)
    power = transition.copy()
    for k in range(rwpe_dim):
        rwpe[:, k] = np.diag(power)
        power = power @ transition

    return rwpe.astype(np.float16)


def _extract_3d_coords(mol, sdf_mol, keep_atom):
    """
    Extract 3D coordinates for atoms that are kept in the returned graph.

    :param mol: RDKit molecule object with hydrogens added (and optional reordering)
    :param sdf_mol: RDKit molecule loaded from the training SDF, or None
    :param keep_atom: boolean mask over mol atoms indicating nodes to retain
    :return: np.ndarray with shape (num_kept_atoms, 3), dtype float16
    """
    if sdf_mol is None:
        return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)

    try:
        if sdf_mol.GetNumConformers() == 0:
            return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)
        match = sdf_mol.GetSubstructMatch(mol)
    except Exception:
        return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)

    if len(match) != mol.GetNumAtoms() or any(m < 0 for m in match):
        return np.zeros((int(keep_atom.sum()), COORD_DIM), dtype=np.float16)

    conf = sdf_mol.GetConformer()
    coords = []
    for smi_idx in range(mol.GetNumAtoms()):
        sdf_idx = int(match[smi_idx])
        pos = conf.GetAtomPosition(sdf_idx)
        coords.append([pos.x, pos.y, pos.z])

    node_coords = np.asarray(coords, dtype=np.float32)
    node_coords = node_coords[keep_atom]
    if node_coords.shape[0] == 0:
        return np.zeros((0, COORD_DIM), dtype=np.float16)

    centroid = node_coords.mean(axis=0, keepdims=True)
    node_coords = node_coords - centroid
    return node_coords.astype(np.float16)


def _rotatable_bond_indices(mol) -> set[int]:
    """Return bond indices of rotatable bonds using an RDKit substructure pattern."""
    if _ROTATABLE_BOND_SMARTS is None:
        return set()
    matches = mol.GetSubstructMatches(_ROTATABLE_BOND_SMARTS)
    bond_indices = set()
    for atom_idx_a, atom_idx_b in matches:
        bond = mol.GetBondBetweenAtoms(atom_idx_a, atom_idx_b)
        if bond is not None:
            bond_indices.add(int(bond.GetIdx()))
    return bond_indices


def _is_rotable_bond(bond, rotatable_bond_indices: set[int] | None = None):
    """Return whether an RDKit bond is rotatable."""
    if rotatable_bond_indices is not None:
        return bond.GetIdx() in rotatable_bond_indices
    return False


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], _non_active_hydrogen_count(atom)),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            safe_index(allowable_features['possible_ring_size_list'],
                       atom.GetOwningMol().GetRingInfo().MinAtomRingSize(atom.GetIdx())),
            ]
    return atom_feature


def bond_to_feature_vector(bond, rotatable_bond_indices: set[int] | None = None):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
                allowable_features['possible_is_rotable_list'].index(
                    _is_rotable_bond(bond, rotatable_bond_indices),
                ),
                safe_index(allowable_features['possible_ring_size_list'],
                           bond.GetOwningMol().GetRingInfo().MinBondRingSize(bond.GetIdx())),
            ]
    return bond_feature


def smiles2graph(smiles_string, removeHs=True, sdf_mol=None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
    except Exception:
        pass  # charges will fall back to 0.0 via _get_gasteiger_charge
    if removeHs:
        keep_atom = _active_hydrogen_mask(mol)
    else:
        keep_atom = np.ones(mol.GetNumAtoms(), dtype=np.bool_)

    keep_idx = np.flatnonzero(keep_atom)
    if len(keep_idx) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_feat = np.empty((0, 5), dtype=np.int64)
        return {
            'edge_index': edge_index,
            'edge_feat': edge_feat,
            'node_feat': np.empty((0, 0), dtype=np.int64),
            'node_embd': np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
            'num_nodes': 0,
        }

    old_to_new = -np.ones(mol.GetNumAtoms(), dtype=np.int32)
    old_to_new[keep_idx] = np.arange(len(keep_idx))
    rotatable_bond_indices = _rotatable_bond_indices(mol)

    # atoms
    atom_feat_list = []
    node_en_list = []
    node_gc_list = []
    for atom in mol.GetAtoms():
        if keep_atom[atom.GetIdx()]:
            atom_feat_list.append(atom_to_feature_vector(atom))
            node_en_list.append(_get_centered_en(atom))
            node_gc_list.append(_get_gasteiger_charge(atom))
    node_coords = _extract_3d_coords(mol, sdf_mol, keep_atom)
    x = np.array(atom_feat_list, dtype = np.int64)
    node_en = np.array(node_en_list, dtype=np.float16).reshape(-1, 1)
    node_gc = np.array(node_gc_list, dtype=np.float16).reshape(-1, 1)
    node_coords = np.asarray(node_coords, dtype=np.float16)

    # bonds
    num_bond_features = 5  # bond type, bond stereo, is_conjugated, is_rotable, ring_size
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_feat_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if not (keep_atom[i] and keep_atom[j]):
                continue

            edge_feature = bond_to_feature_vector(bond, rotatable_bond_indices)

            ni, nj = old_to_new[i], old_to_new[j]
            # add edges in both directions
            edges_list.append((ni, nj))
            edge_feat_list.append(edge_feature)
            edges_list.append((nj, ni))
            edge_feat_list.append(edge_feature)

        if len(edges_list) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_feat = np.empty((0, num_bond_features), dtype=np.int64)
        else:
            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_feat: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_feat = np.array(edge_feat_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_feat = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_feat
    graph['node_feat'] = x
    graph['node_embd'] = np.concatenate(
        [_compute_rwpe(len(x), edge_index, RWPE_DIM), node_en, node_gc, node_coords], axis=1
    )
    graph['num_nodes'] = len(x)

    return graph


def _concat_graph_blocks(graphs: list):
    """Concatenate node/edge arrays and build boundary pointers."""
    node_ptr = [0]
    edge_ptr = [0]
    node_feat_list = []
    node_embd_list = []
    edge_feat_list = []
    edge_indices = []

    node_feat_dim = 0
    node_embd_dim = 0
    edge_feat_dim = 0
    for graph in graphs:
        node_feat = np.asarray(graph['node_feat'])
        node_embd = np.asarray(graph.get('node_embd', np.zeros((node_feat.shape[0], NODE_CONTINUOUS_DIM), dtype=np.float16)))
        edge_feat = np.asarray(graph['edge_feat'])

        if node_feat.size > 0 and node_feat.ndim == 2:
            node_feat_dim = node_feat.shape[1]
        if node_embd.size > 0 and node_embd.ndim == 2:
            node_embd_dim = node_embd.shape[1]
        if edge_feat.size > 0 and edge_feat.ndim == 2:
            edge_feat_dim = edge_feat.shape[1]
        if node_feat_dim > 0 and node_embd_dim > 0 and edge_feat_dim > 0:
            break

    if node_feat_dim == 0:
        for graph in graphs:
            edge_feat = np.asarray(graph['edge_feat'])
            if edge_feat.size > 0 and edge_feat.ndim == 2:
                edge_feat_dim = edge_feat.shape[1]
                break
    if node_embd_dim == 0:
        node_embd_dim = NODE_CONTINUOUS_DIM

    for graph in graphs:
        node_feat = np.asarray(graph['node_feat'])
        node_embd = np.asarray(graph.get('node_embd', np.zeros((node_feat.shape[0], node_embd_dim), dtype=np.float16)))
        edge_feat = np.asarray(graph['edge_feat'])
        edge_index = np.asarray(graph['edge_index'])

        if node_feat.size == 0:
            node_feat = np.zeros((0, node_feat_dim), dtype=np.int8)
        if node_embd.size == 0:
            node_embd = np.zeros((0, node_embd_dim), dtype=np.float16)
        if edge_feat.size == 0:
            edge_feat = np.zeros((0, edge_feat_dim), dtype=np.int8)

        node_feat_list.append(node_feat)
        node_embd_list.append(node_embd)
        edge_feat_list.append(edge_feat)
        num_nodes = node_feat.shape[0]
        num_edges = edge_index.shape[1] if edge_index.size else 0

        if num_edges > 0:
            edge_indices.append(np.asarray(edge_index))
        else:
            edge_indices.append(np.zeros((2, 0), dtype=np.int32))

        node_ptr.append(node_ptr[-1] + num_nodes)
        edge_ptr.append(edge_ptr[-1] + num_edges)

    if node_feat_list:
        node_feat_arr = np.concatenate(node_feat_list, axis=0)
    else:
        node_feat_arr = np.zeros((0, 0))

    if edge_feat_list:
        edge_feat_arr = np.concatenate(edge_feat_list, axis=0)
    else:
        edge_feat_arr = np.zeros((0, 0))

    if node_embd_list:
        node_embd_arr = np.concatenate(node_embd_list, axis=0)
    else:
        node_embd_arr = np.zeros((0, node_embd_dim), dtype=np.float16)

    if edge_indices:
        edge_index = np.concatenate(edge_indices, axis=1) if edge_indices else np.zeros((2, 0), dtype=np.int32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int32)

    return (
        np.asarray(node_feat_arr, dtype=np.int8),
        np.asarray(node_embd_arr, dtype=np.float16),
        np.asarray(edge_feat_arr, dtype=np.int8),
        np.asarray(edge_index, dtype=np.int8),
        np.asarray(node_ptr, dtype=np.int32),
        np.asarray(edge_ptr, dtype=np.int32),
    )


def _save_hdf5(path: str, graphs: list, labels: np.ndarray):
    """Persist concatenated graph tensors and labels in HDF5 format."""
    node_feat, node_embd, edge_feat, edge_index, node_ptr, edge_ptr = _concat_graph_blocks(graphs)

    os.makedirs(osp.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('labels', data=np.asarray(labels))
        f.create_dataset('node_feat', data=node_feat)
        f.create_dataset('node_embd', data=node_embd)
        f.create_dataset('edge_feat', data=edge_feat)
        f.create_dataset('edge_index', data=edge_index)
        f.create_dataset('node_ptr', data=node_ptr)
        f.create_dataset('edge_ptr', data=edge_ptr)


def _load_hdf5(path: str):
    """Load concatenated graph arrays and boundary pointers from HDF5."""
    with h5py.File(path, 'r') as f:
        labels = np.asarray(f['labels'][()], dtype=np.float32)
        node_feat = np.asarray(f['node_feat'][()], dtype=np.int32)
        node_embd = np.asarray(f['node_embd'][()], dtype=np.float32)
        edge_feat = np.asarray(f['edge_feat'][()], dtype=np.int32)
        edge_index = np.asarray(f['edge_index'][()], dtype=np.int32)
        node_ptr = np.asarray(f['node_ptr'][()], dtype=np.int32)
        edge_ptr = np.asarray(f['edge_ptr'][()], dtype=np.int32)

    return labels, node_feat, node_embd, edge_feat, edge_index, node_ptr, edge_ptr


class PCQM4Mv2Dataset(object):
    def __init__(self, root = 'dataset', smiles2graph = smiles2graph, only_smiles=False):
        '''
        Library-agnostic PCQM4Mv2 dataset object
            - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
            - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                * The default smiles2graph requires rdkit to be installed
            - only_smiles (bool): If this is true, we directly return the SMILES string in our __get_item__, without converting it into a graph.
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.only_smiles = only_smiles
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 2
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')

        super(PCQM4Mv2Dataset, self).__init__()

        # Prepare everything.
        # download if there is no raw file
        # preprocess if there is no processed file
        # load data if processed file is found.
        if self.only_smiles:
            self.prepare_smiles()
        else:
            self.prepare_graph()

    def download(self):
        from ogb.utils.url import decide_download, download_url, extract_zip

        raw_dir = osp.join(self.folder, 'raw')
        if osp.exists(osp.join(raw_dir, 'data.csv.gz')):
            print('Raw file already exists, skipping download.')
            return

        archive_path = osp.join(self.original_root, self.url.rpartition('/')[2])
        if osp.exists(archive_path) and osp.getsize(archive_path) > 0:
            print('Using existing archive', osp.basename(archive_path))
            extract_zip(archive_path, self.original_root)
            return

        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Download cancelled.')
            exit(-1)

    def prepare_smiles(self):
        raw_dir = osp.join(self.folder, 'raw')
        if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
            self.download()

        data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles'].values
        homolumogap_list = data_df['homolumogap'].values
        self.graphs = list(smiles_list)
        self.labels = homolumogap_list

    def prepare_graph(self):
        processed_dir = osp.join(self.folder, 'processed')
        raw_dir = osp.join(self.folder, 'raw')
        pre_processed_file_path = osp.join(processed_dir, 'data_processed.h5')

        if osp.exists(pre_processed_file_path):        
            # if pre-processed file already exists
            (
                self.labels,
                self.node_feat,
                self.node_embd,
                self.edge_feat,
                self.edge_index,
                self.node_ptr,
                self.edge_ptr,
            ) = _load_hdf5(pre_processed_file_path)
            self.graphs = None
        
        else:
            # if pre-processed file does not exist
            os.makedirs(processed_dir, exist_ok=True)
            
            if not osp.exists(osp.join(raw_dir, 'data.csv.gz')):
                self.download()

            data_df = pd.read_csv(osp.join(raw_dir, 'data.csv.gz'))
            smiles_list = data_df['smiles']
            homolumogap_list = data_df['homolumogap']
            sdf_path = osp.join(raw_dir, 'pcqm4m-v2-train.sdf')
            sdf_suppl = None
            sdf_len = -1
            if osp.exists(sdf_path):
                RDLogger.DisableLog('rdApp.*')
                sdf_suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
                if sdf_suppl is not None:
                    sdf_len = len(sdf_suppl)
                    print(f'Loaded SDF with {sdf_len} molecules from {sdf_path}')
                else:
                    print(f'Warning: failed to create SDMolSupplier from {sdf_path}, continuing with no 3D coords.')
            else:
                print(f'Warning: {sdf_path} not found, continuing with no 3D coords.')

            print('Converting SMILES strings into graphs...')
            self.graphs = []
            self.labels = []
            for i in tqdm(range(len(smiles_list))):

                smiles = smiles_list[i]
                homolumogap = homolumogap_list[i]
                sdf_mol = sdf_suppl[i] if (sdf_suppl is not None and i < sdf_len) else None
                try:
                    graph = self.smiles2graph(smiles, sdf_mol=sdf_mol)
                except AttributeError as exc:
                    if "'NoneType' object has no attribute 'GetAtoms'" in str(exc):
                        graph = {
                            'node_feat': np.zeros((0, 0), dtype=np.int8),
                            'node_embd': np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
                            'edge_feat': np.zeros((0, 0), dtype=np.int8),
                            'edge_index': np.zeros((2, 0), dtype=np.int32),
                            'num_nodes': 0,
                        }
                        homolumogap = -1
                    else:
                        raise
                else:
                    if graph is None:
                        graph = {
                            'node_feat': np.zeros((0, 0), dtype=np.int8),
                            'node_embd': np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
                            'edge_feat': np.zeros((0, 0), dtype=np.int8),
                            'edge_index': np.zeros((2, 0), dtype=np.int32),
                            'num_nodes': 0,
                        }
                        homolumogap = -1
                    else:
                        assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
                        assert(len(graph['node_feat']) == graph['num_nodes'])

                self.graphs.append(graph)
                self.labels.append(homolumogap)

            self.labels = np.array(self.labels)
            print(self.labels)

            # double-check prediction target
            split_dict = self.get_idx_split()
            assert(all([not np.isnan(self.labels[i]) or self.labels[i] == -1 for i in split_dict['train']]))
            assert(all([not np.isnan(self.labels[i]) or self.labels[i] == -1 for i in split_dict['valid']]))
            assert(all([(np.isnan(self.labels[i]) or self.labels[i] == -1) for i in split_dict['test-dev']]))
            assert(all([(np.isnan(self.labels[i]) or self.labels[i] == -1) for i in split_dict['test-challenge']]))

            print('Saving...')
            _save_hdf5(pre_processed_file_path, self.graphs, self.labels)
            (
                self.labels,
                self.node_feat,
                self.node_embd,
                self.edge_feat,
                self.edge_index,
                self.node_ptr,
                self.edge_ptr,
            ) = _load_hdf5(pre_processed_file_path)
            self.graphs = None

    def get_idx_split(self):
        split_dict_path = osp.join(self.folder, 'split_dict.h5')
        if osp.exists(split_dict_path):
            with h5py.File(split_dict_path, 'r') as f:
                return {key: np.asarray(f[key][()]) for key in f.keys()}

        raise FileNotFoundError(
            f'Expected split dictionary at {split_dict_path}, but it was not found.'
            ' This codebase is HDF5-only and does not read split_dict.pt.'
        )

    def __getitem__(self, idx):
        '''Get datapoint with index'''

        if isinstance(idx, (int, np.integer)):
            if self.graphs is None:
                node_start = int(self.node_ptr[idx])
                node_end = int(self.node_ptr[idx + 1])
                edge_start = int(self.edge_ptr[idx])
                edge_end = int(self.edge_ptr[idx + 1])

                return (
                    {
                        'node_feat': self.node_feat[node_start:node_end],
                        'node_embd': self.node_embd[node_start:node_end],
                        'edge_feat': self.edge_feat[edge_start:edge_end],
                        'edge_index': self.edge_index[:, edge_start:edge_end],
                        'num_nodes': node_end - node_start,
                        'num_edges': edge_end - edge_start,
                    },
                    self.labels[idx],
                )

            return self.graphs[idx], self.labels[idx]

        raise IndexError(
            'Only integer is valid index (got {}).'.format(type(idx).__name__))

    def __len__(self):
        '''Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        '''
        return len(self.labels)

    def __repr__(self):  # pragma: no cover
        return '{}({})'.format(self.__class__.__name__, len(self))


class PCQM4Mv2Evaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4Mv2 dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray of shape (num_graphs,)
            y_pred: numpy.ndarray of shape (num_graphs,)
            y_true and y_pred need to be numpy.ndarrays with same shape
        '''
        assert('y_pred' in input_dict)
        assert('y_true' in input_dict)

        y_pred, y_true = input_dict['y_pred'], input_dict['y_true']

        assert((isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray)))
        assert(y_true.shape == y_pred.shape)
        assert(len(y_true.shape) == 1)

        return {'mae': float(np.mean(np.absolute(y_pred - y_true)))}

    def save_test_submission(self, input_dict: Dict, dir_path: str, mode: str):
        '''
            save test submission file at dir_path
        '''
        assert('y_pred' in input_dict)
        assert mode in ['test-dev', 'test-challenge']

        y_pred = input_dict['y_pred']

        if mode == 'test-dev':
            filename = osp.join(dir_path, 'y_pred_pcqm4m-v2_test-dev')
            assert(y_pred.shape == (147037,))
        elif mode == 'test-challenge':
            filename = osp.join(dir_path, 'y_pred_pcqm4m-v2_test-challenge')
            assert(y_pred.shape == (147432,))

        assert(isinstance(filename, str))
        assert(isinstance(y_pred, np.ndarray))

        if not osp.exists(dir_path):
            os.makedirs(dir_path)

        y_pred = y_pred.astype(np.float32)
        np.savez_compressed(filename, y_pred = y_pred)


if __name__ == '__main__':
    dataset = PCQM4Mv2Dataset()
    print(dataset)
    print(dataset[1234])

    split_dict = dataset.get_idx_split()
    print(split_dict['train'].shape)
    print(split_dict['valid'].shape)
    print('-----------------')
    print(split_dict['test-dev'].shape)
    print(split_dict['test-challenge'].shape)


    evaluator = PCQM4Mv2Evaluator()
    y_true = np.random.randn(100)
    y_pred = np.random.randn(100)
    result = evaluator.eval({'y_true': y_true, 'y_pred': y_pred})
    print(result)

    print(len(split_dict['test-dev']))
    print(len(split_dict['test-challenge']))

    y_pred = np.random.randn(len(split_dict['test-dev']))
    evaluator.save_test_submission({'y_pred': y_pred}, 'results',mode = 'test-dev')

    y_pred = np.random.randn(len(split_dict['test-challenge']))
    evaluator.save_test_submission({'y_pred': y_pred}, 'results',mode = 'test-challenge')
