"""
Classes to generate a Molecule instance from GROMACS input
"""
from typing import Union
from pathlib import Path

from grappa.data import Molecule

class GromacsMolecule(Molecule):

    @classmethod
    def from_top(cls, top_path:Union[Path,str]):
        raise NotImplementedError

    def from_kimmdy(cls, kimmdylist):
        pass