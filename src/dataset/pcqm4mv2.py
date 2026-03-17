from typing import Dict

import os
import os.path as osp

import h5py
from rdkit import Chem
from rdkit import RDLogger
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from .features import NODE_CONTINUOUS_DIM
from .graph import mol_to_graph
from .hdf5 import load_graphs, save_graphs


class PCQM4Mv2Dataset(object):
    def __init__(self, root='dataset', smiles2graph=mol_to_graph, only_smiles=False):
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
            (
                self.labels,
                self.node_feat,
                self.node_embd,
                self.edge_feat,
                self.edge_index,
                self.node_ptr,
                self.edge_ptr,
                self.edge_index_2hop,
                self.edge_feat_2hop,
                self.edge_ptr_2hop,
                self.edge_index_3hop,
                self.edge_feat_3hop,
                self.edge_ptr_3hop,
                self.edge_index_4hop,
                self.edge_feat_4hop,
                self.edge_ptr_4hop,
            ) = load_graphs(pre_processed_file_path)
            self.graphs = None

        else:
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
                            'node_feat': np.zeros((0, 0), dtype=np.uint8),
                            'node_embd': np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
                            'edge_feat': np.zeros((0, 0), dtype=np.uint8),
                            'edge_index': np.zeros((2, 0), dtype=np.int32),
                            'edge_index_2hop': np.zeros((2, 0), dtype=np.int32),
                            'edge_feat_2hop': np.zeros((0, 2), dtype=np.uint8),
                            'edge_index_3hop': np.zeros((2, 0), dtype=np.int32),
                            'edge_feat_3hop': np.zeros((0, 3), dtype=np.uint8),
                            'edge_index_4hop': np.zeros((2, 0), dtype=np.int32),
                            'edge_feat_4hop': np.zeros((0, 4), dtype=np.uint8),
                            'num_nodes': 0,
                        }
                        homolumogap = -1
                    else:
                        raise
                else:
                    if graph is None:
                        graph = {
                            'node_feat': np.zeros((0, 0), dtype=np.uint8),
                            'node_embd': np.zeros((0, NODE_CONTINUOUS_DIM), dtype=np.float16),
                            'edge_feat': np.zeros((0, 0), dtype=np.uint8),
                            'edge_index': np.zeros((2, 0), dtype=np.int32),
                            'edge_index_2hop': np.zeros((2, 0), dtype=np.int32),
                            'edge_feat_2hop': np.zeros((0, 2), dtype=np.uint8),
                            'edge_index_3hop': np.zeros((2, 0), dtype=np.int32),
                            'edge_feat_3hop': np.zeros((0, 3), dtype=np.uint8),
                            'edge_index_4hop': np.zeros((2, 0), dtype=np.int32),
                            'edge_feat_4hop': np.zeros((0, 4), dtype=np.uint8),
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

            split_dict = self.get_idx_split()
            assert(all([not np.isnan(self.labels[i]) or self.labels[i] == -1 for i in split_dict['train']]))
            assert(all([not np.isnan(self.labels[i]) or self.labels[i] == -1 for i in split_dict['valid']]))
            assert(all([(np.isnan(self.labels[i]) or self.labels[i] == -1) for i in split_dict['test-dev']]))
            assert(all([(np.isnan(self.labels[i]) or self.labels[i] == -1) for i in split_dict['test-challenge']]))

            print('Saving...')
            save_graphs(pre_processed_file_path, self.graphs, self.labels)
            (
                self.labels,
                self.node_feat,
                self.node_embd,
                self.edge_feat,
                self.edge_index,
                self.node_ptr,
                self.edge_ptr,
                self.edge_index_2hop,
                self.edge_feat_2hop,
                self.edge_ptr_2hop,
                self.edge_index_3hop,
                self.edge_feat_3hop,
                self.edge_ptr_3hop,
                self.edge_index_4hop,
                self.edge_feat_4hop,
                self.edge_ptr_4hop,
            ) = load_graphs(pre_processed_file_path)
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
                edge_start_2hop = int(self.edge_ptr_2hop[idx])
                edge_end_2hop = int(self.edge_ptr_2hop[idx + 1])
                edge_start_3hop = int(self.edge_ptr_3hop[idx])
                edge_end_3hop = int(self.edge_ptr_3hop[idx + 1])
                edge_start_4hop = int(self.edge_ptr_4hop[idx])
                edge_end_4hop = int(self.edge_ptr_4hop[idx + 1])

                return (
                    {
                        'node_feat': self.node_feat[node_start:node_end],
                        'node_embd': self.node_embd[node_start:node_end],
                        'edge_feat': self.edge_feat[edge_start:edge_end],
                        'edge_index': self.edge_index[:, edge_start:edge_end],
                        'edge_index_2hop': self.edge_index_2hop[:, edge_start_2hop:edge_end_2hop],
                        'edge_feat_2hop': self.edge_feat_2hop[edge_start_2hop:edge_end_2hop],
                        'edge_index_3hop': self.edge_index_3hop[:, edge_start_3hop:edge_end_3hop],
                        'edge_feat_3hop': self.edge_feat_3hop[edge_start_3hop:edge_end_3hop],
                        'edge_index_4hop': self.edge_index_4hop[:, edge_start_4hop:edge_end_4hop],
                        'edge_feat_4hop': self.edge_feat_4hop[edge_start_4hop:edge_end_4hop],
                        'num_nodes': node_end - node_start,
                        'num_edges': edge_end - edge_start,
                        'num_2hop_edges': edge_end_2hop - edge_start_2hop,
                        'num_3hop_edges': edge_end_3hop - edge_start_3hop,
                        'num_4hop_edges': edge_end_4hop - edge_start_4hop,
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
