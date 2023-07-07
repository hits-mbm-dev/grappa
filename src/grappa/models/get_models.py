
from grappa.models import readout
from grappa.models import gated_torsion
from grappa.models import old_gated_torsion

from grappa.models.readout import get_default_statistics
from grappa.models.graph_attention_model import Representation
from grappa.models.old_graph_model import Representation as old_Representation
from grappa.models import old_readout
import torch
from typing import Union, List, Tuple


def get_readout(statistics, rep_feats=500, between_feats=1000, old=False):

    if old:
        readout_module = old_readout
        torsion_module = old_gated_torsion
    else:
        readout_module = readout
        torsion_module = gated_torsion


    bond_angle = torch.nn.Sequential(
        readout_module.WriteBondParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics),
        readout_module.WriteAngleParameters(rep_feats=rep_feats, between_feats=between_feats, stat_dict=statistics)
    )

    torsion = torsion_module.GatedTorsion(rep_feats=rep_feats, between_feats=between_feats, improper=False)
    improper = torsion_module.GatedTorsion(rep_feats=rep_feats, between_feats=between_feats, improper=True)

    model = bond_angle

    model.add_module("torsion", torsion)
    model.add_module("improper", improper)

    return model



def get_full_model(statistics=None, rep_feats=512, between_feats=1024, n_res=5, in_feat_name:Union[str,List[str]]=["atomic_number", "residue", "in_ring", "mass", "degree", "formal_charge"], bonus_features=[], bonus_dims=[], old=False, n_heads=6):
    
    if statistics is None:
        statistics = get_default_statistics()

    if old:
        representation = old_Representation(h_feats=between_feats, out_feats=rep_feats, n_residuals=n_res, n_conv=1, in_feat_name=in_feat_name, bonus_features=bonus_features, bonus_dims=bonus_dims)
    else:
        representation = Representation(h_feats=between_feats, out_feats=rep_feats, n_conv=n_res, in_feat_name=in_feat_name, bonus_features=bonus_features, bonus_dims=bonus_dims, n_heads=n_heads)


    readout = get_readout(statistics, rep_feats=rep_feats, between_feats=between_feats, old=old)

    model = torch.nn.Sequential(
        representation,
        readout
    )

    return model