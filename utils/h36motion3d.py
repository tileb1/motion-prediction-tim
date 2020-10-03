from torch.utils.data import Dataset
import numpy as np
from utils import data_utils
import torch
from utils.model import IdentityAutoencoder


class H36motion3D(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, split=0, sample_rate=2,
                 autoencoder=IdentityAutoencoder(), subset=False, treat_subj5_differently=True):
        """
        :param split: 0 train, 1 testing, 2 validation
        """
        # Note: the default autoencoder is an indentity mapping which is what is used in the paper
        self.path_to_data = path_to_data
        self.split = split

        subs = np.array([[1, 6, 7, 8, 9], [5], [11]], dtype=object)
        acts = data_utils.define_actions(actions)

        if subset:
            subs = np.array([[1], [5], [11]], dtype=object)
            acts = ['walking']

        subjs = subs[split]
        all_seqs, dim_ignore, dim_used = data_utils.load_data_3d(path_to_data, subjs, acts, sample_rate,
                                                                 input_n + output_n,
                                                                 treat_subj5_differently=treat_subj5_differently)
        self.all_seqs = all_seqs
        self.dim_used = dim_used

        # (nb_total_seq, len_seq, nb_joints)
        all_seqs = torch.from_numpy(all_seqs[:, :, dim_used]).float()

        # (nb_total_seq, nb_joints, hidden_dim)
        self.all_seqs_encoded = autoencoder(all_seqs.transpose(2, 1))[1]

    def __len__(self):
        return self.all_seqs_encoded.shape[0]

    def __getitem__(self, item):
        return self.all_seqs_encoded[item], self.all_seqs[item]
