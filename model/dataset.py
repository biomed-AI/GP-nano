import os
import logging
import pickle
import torch
import dgl
import pandas as pd


from typing import List
from dgl.data import DGLDataset, download, check_sha1
#import dgl.heterograph
import torch.multiprocessing
import warnings
warnings.simplefilter('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')


def dgl_picp_collate(batch):
    """Assemble a protein complex dictionary batch into two large batched DGLGraphs and a batched labels tensor."""
    complex_dicts = []
    labels= []
    for item in batch:
        complex_dicts.append(item[0])
        labels.append(item[1])
    batched_graph = dgl.batch([complex_dict['graph'] for complex_dict in complex_dicts])
    labels = torch.tensor([item.detach().numpy() for item in labels])
    return batched_graph,labels


class NanobodyDataset(DGLDataset):
    def __init__(self,
                 mode='test',
                 raw_dir="yourpath/graph",
                 nuv_residue=True,
                 nuv_angle=False,
                 ):
        assert mode in ['train', 'val', 'test']
        #raw_dir1 = '/home/zhouxiaolong/data/antibody'
        pdb_file = open(os.path.join(raw_dir, "{}.txt".format(mode)), "r")
        pdb_ids = [pdb.strip() for pdb in pdb_file.readlines()]



        self.mode = mode
        self.root = raw_dir
        self.final_dir = f"{os.sep}".join(self.root.split(os.sep)[:-1])
        self.processed_dir = os.path.join(self.final_dir, 'processed')
        self.nuv_residue = nuv_residue
        self.nuv_angle = nuv_angle

        self.filenames_frame = []
        for pdb_id in pdb_ids:
            if pdb_id[-1]=='1':
                self.filenames_frame.append(os.path.join(raw_dir, "high", pdb_id[:-2]+'.pkl'))
            elif pdb_id[-1]=='0':
                self.filenames_frame.append(os.path.join(raw_dir, "low", pdb_id[:-2]+'.pkl'))
        self.label_frame = []
        for pdb_id in pdb_ids:
            if pdb_id[-1]=='1':
                l = [1]         
            else:
                l = [0]
            self.label_frame.append(l)
        self.label_frame = torch.Tensor(self.label_frame)

        super(NanobodyDataset, self).__init__(name='Nanobody',
                                              raw_dir=raw_dir)
        logging.info(f"Loaded Nanobody-Plus {mode}-set, source: {self.processed_dir}, length: {len(self)}")

    def download(self):
        """Download and extract a pre-packaged version of the raw pairs if 'self.raw_dir' is not already populated."""
        # Path to store the file
        gz_file_path = os.path.join(os.path.join(*self.raw_dir.split(os.sep)[:-1]), 'final_raw_casp_capri.tar.gz')

        # Download file
        download(self.url, path=gz_file_path)

        # Check SHA-1
        if not check_sha1(gz_file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(gz_file_path))

        # Remove existing raw directory to make way for the new archive to be extracted
        if os.path.exists(self.raw_dir):
            os.removedirs(self.raw_dir)

        # Extract archive to parent directory of `self.raw_dir`
        self._extract_gz(gz_file_path, os.path.join(*self.raw_dir.split(os.sep)[:-1]))


    def process(self):
        """Process each protein complex into a testing-ready dictionary representing both structures."""
        if self.process_complexes:
            # Ensure the directory of processed complexes is already created
            os.makedirs(self.processed_dir, exist_ok=True)
            # Process each unprocessed protein complex
            for (i, raw_path) in self.filenames_frame.iterrows():
                raw_filepath = os.path.join(self.raw_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                processed_filepath = os.path.join(self.processed_dir, f'{os.path.splitext(raw_path[0])[0]}.dill')
                if not os.path.exists(processed_filepath):
                    raise NotImplementedError

    def has_cache(self):
        return True

    def __getitem__(self, idx):
        r""" Get feature dictionary by index of complex.

        Parameters
        ----------
        idx: int

        Returns
        -------
        :class:`dict`

    - ``complex['graph1']:`` DGLGraph (of length M) containing each of the first graph's encoded node and edge features
    - ``complex['graph2']:`` DGLGraph (of length N) containing each of the second graph's encoded node and edge features
    - ``complex['examples']:`` PyTorch Tensor (of shape (M x N) x 3) containing the labels for inter-graph node pairs
    - ``complex['complex']:`` Python string describing the complex's code and original pdb filename
    - ``complex['filepath']:`` Python string describing the complex's filepath
        """
        # Assemble filepath of processed protein complex
        processed_filepath = self.filenames_frame[idx]
        pdb_id = processed_filepath.split("/")[-1].split(".")[0]
        # Load in processed complex
        with open(processed_filepath, 'rb') as f:
            processed_complex = pickle.load(f)
        label = self.label_frame[idx]
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph = processed_complex["graph"]
        #graph = graph.to(device)

        # if self.nuv_residue:
        #     graph1.ndata['f'] = torch.cat([graph1.ndata["f"], graph1.ndata["nuv"].flatten(1)], dim=-1)
        #     graph2.ndata['f'] = torch.cat([graph2.ndata["f"], graph2.ndata["nuv"].flatten(1)], dim=-1)

        if self.nuv_angle:
            normal = graph.ndata["nuv"][:, 0, :]
            graph.edata['f'] = torch.cat([
                graph.edata["f"][:, :-1],
                (normal[graph.edges()[0]] * normal[graph.edges()[1]]).sum(-1, keepdim=True)
            ], dim=-1)

        return processed_complex,label

    def __len__(self) -> int:
        r"""Number of graph batches in the dataset."""
        return len(self.filenames_frame)

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each inter-chain residue pair in each complex."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        return 113 if not self.nuv_residue else 113 + 9

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 28

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir

    @property
    def url(self) -> str:
        """URL with which to download TAR archive of preprocessed pairs."""
        return 'https://zenodo.org/record/6299835/files/final_processed_casp_capri.tar.gz'


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    torch.multiprocessing.set_sharing_strategy("file_system")


    for m in ["train"]:  # , "val", "test"
        dataset = NanobodyDataset(mode=m)
        dataloader = DataLoader(dataset,batch_size=1,shuffle=False,pin_memory=False,num_workers=64,collate_fn=dgl_picp_collate)
        
        for graph,label in dataloader:
            # print(type(graph.ndata['f']))
            print(graph.ndata['f'].shape)
            print(graph.ndata['nuv'][:,0].shape)
            feat = torch.concat((graph.ndata['f'],graph.ndata['nuv'][:,0]),dim=1)
            print(feat.shape)
            break
        break
# Graph(num_nodes=117, num_edges=2340,
#       ndata_schemes={'f': Scheme(shape=(113,), dtype=torch.float32), 'x': Scheme(shape=(3,), dtype=torch.float32), 'nuv': Scheme(shape=(3, 3), dtype=torch.float32)}
#       edata_schemes={'f': Scheme(shape=(28,), dtype=torch.float32), 'src_nbr_e_ids': Scheme(shape=(2,), dtype=torch.int64), 'dst_nbr_e_ids': Scheme(shape=(2,), dtype=torch.int64)})
# tensor([[1.]])

            


      