# GP-nano
GP-nano: a geometric graph network for nanobody polyreactivity prediciton

Nanobodies are emerging therapeutic antibodies with more simple structure, which can target antigen surfaces and tissue types not accessible to conventional antibodies. However, nanobodies exhibit polyreactivity, binding non-specifically to off-target proteins and other biomolecules. This uncertainty can affect the drug development process and pose significant challenges in clinical development. Existing computational polyreactivity prediction methods focus solely on the sequence or fail to fully utilize structural information. In this study, we propose GP-nano, a geometric graph network based model for nanobody polyreactivity using predictive structure. GP-nano starts from sequences, predicts protein structures using ESMfold, and fully utilizes structural geometric information via graph networks. GP-nano can accurately classify the polyreactivity of nanobody sequences (AUC=0.91). To demonstrate GP-nano's generalizability, we also trained and tested it on monoclonal antibodies (mAbs) dataset. GP-nano outperforms the best methods on both datasets, indicating the contribution of structural information and geometric features to antibody polyreactivity prediction.



![image](https://github.com/biomed-AI/GP-nano/blob/main/main_graph.jpg)

## Predict Portein Structure
First we collect protein sequences and use ESMFold(https://github.com/facebookresearch/esm).


## Feature generation and graph construction
Second we need create conda environment with python3.8 and download the following package:
1. pytorch
2. dgl
3. numpy
4. pandas
5. pyg
6. pykeops
7. biopython
Then, we use build_graph_nanobody.py to get feature of every antibody and construct every dgl graph.

## Train and test
Finally, we use the dgl graph to train and test model in Model Folder.
