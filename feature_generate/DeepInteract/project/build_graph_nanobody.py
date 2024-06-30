import os
import logging
import shutil
import pickle

import atom3.pair as pair
import atom3.neighbors as nb
import numpy as np
import torch

from atom3.structure import get_ca_pos_from_residues, get_ca_pos_from_atoms
from tqdm import tqdm

from project.utils.deepinteract_utils import convert_input_pdb_files_to_pair, convert_df_to_dgl_graph

from multiprocessing import Pool
import warnings

warnings.filterwarnings('ignore')


def get_data(file, pdb_id):
    if os.path.exists("~/data/nanobody/graph/{}/{}.pkl".format(file, pdb_id)):
        return
    input_id = "Input"
    try:
        shutil.rmtree(os.path.join('datasets', input_id))
    except FileNotFoundError:
        pass
    left_pdb_filepath = "~/data/nanobody/{}/{}.pdb".format(file, pdb_id)
    right_pdb_filepath = "~/data/nanobody/{}/{}.pdb".format(file, pdb_id)
    input_dataset_dir = os.path.join('datasets', input_id)
    psaia_dir = '~/Programs/PSAIA_1.0_source/bin/linux/psa'
    psaia_config = 'datasets/builder/psaia_config_file_input.txt'
    # hhsuite_db = '~/Data/Databases/small_bfd/bfd-first_non_consensus_sequences.fasta'
    # hhsuite_db = '~/Data/Databases/uniclust30/uniclust30_2018_08/uniclust30_2018_08'
    hhsuite_db = '~/Data/Databases/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
    input_pair = convert_input_pdb_files_to_pair(left_pdb_filepath, right_pdb_filepath,
                                                 input_dataset_dir, psaia_dir,
                                                 psaia_config, hhsuite_db, fast=True)
    lb_df = input_pair.df0
    rb_df = input_pair.df1
    nb_fn = nb.build_get_neighbors("non_heavy_res", 6)
    lres, rres = nb_fn(lb_df, rb_df)
    ldf, rdf = lb_df, rb_df
    lpos = get_ca_pos_from_residues(ldf, lres)
    rpos = get_ca_pos_from_residues(rdf, rres)
    pos_idx, neg_idx = pair._get_residue_positions(ldf, lpos, rdf, rpos, False)
    ca0 = lb_df[lb_df['atom_name'] == 'CA'].reset_index()
    ca1 = rb_df[rb_df['atom_name'] == 'CA'].reset_index()
    label = []
    for raw in pos_idx:
        i, j = raw
        n_i = int(ca0[ca0["index"] == i].index.values)
        n_j = int(ca1[ca1["index"] == j].index.values)
        label.append((n_i, n_j))
    knn = 20
    geo_nbrhd_size = 2
    self_loops = True
    # Convert the input DataFrame into its DGLGraph representations, using all atoms to generate geometric features
    graph1 = convert_df_to_dgl_graph(input_pair.df0, left_pdb_filepath, knn, geo_nbrhd_size, self_loops)
    # graph2 = convert_df_to_dgl_graph(input_pair.df1, right_pdb_filepath, knn, geo_nbrhd_size, self_loops)
    data = {
        'graph': graph1,
    }
    with open("~/data/nanobody/graph/{}/{}.pkl".format(file, pdb_id), "wb") as f:
        pickle.dump(data, f)


def try_log(h_l, pdb_id, func=get_data):
    try:
        func(h_l, pdb_id)
    except Exception as e:
        print(e)
        f = open("error_log.txt", "a")
        f.write("{}/{}".format(h_l, pdb_id))
        f.write("\n")
        f.close()


if __name__ == "__main__":
    open("error_log.txt", "w")
    pdb_ids = []
    h_l = "low"
    for root, _, files in os.walk("~/data/nanobody/{}".format(h_l)):
        for f in files:
            if f[-3:] == "pdb":
                pdb_ids.append(f[:-4])
    # get_data(pdb_ids[0].strip())
    for pdb_id in tqdm(pdb_ids):
        try_log(h_l, pdb_id.strip())
    # pdb_list = []
    # for pdb_id in pdb_ids:
    #     pdb_list.append(pdb_id.strip())
    # with Pool(32) as p:
    #     p.map_async(try_log, pdb_list)
    #     p.close()
    #     p.join()
