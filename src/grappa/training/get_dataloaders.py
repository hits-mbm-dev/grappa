from grappa.data import Dataset, GraphDataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Union


def get_dataloaders(datasets:List[Union[Path, str, Dataset]], conf_strategy:str='mean', train_batch_size:int=1, val_batch_size:int=1, test_batch_size:int=1, train_loader_workers:int=1, val_loader_workers:int=2, test_loader_workers:int=2, pin_memory:bool=True, splitpath:Path=None, partition:Union[Tuple[float,float,float], Dict[str, Tuple[float, float, float]]]=(0.8,0.1,0.1))->Tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    """
    This function returns train, validation, and test dataloaders for a given list of datasets.

    Args:
        datasets (List[Path]): List of paths to the datasets.
        conf_strategy (str, optional): Strategy for configuration. Defaults to 'mean'.
        train_batch_size (int, optional): Batch size for the training dataloader. Defaults to 1.
        val_batch_size (int, optional): Batch size for the validation dataloader. Defaults to 1.
        test_batch_size (int, optional): Batch size for the test dataloader. Defaults to 1.
        train_loader_workers (int, optional): Number of worker processes for the training dataloader. Defaults to 1.
        val_loader_workers (int, optional): Number of worker processes for the validation dataloader. Defaults to 2.
        test_loader_workers (int, optional): Number of worker processes for the test dataloader. Defaults to 2.
        pin_memory (bool, optional): Whether to pin memory for the dataloaders. Defaults to True.
        splitpath (Path, optional): Path to the split file. If provided, the function will load the split from this file. If not, it will generate a new split. Defaults to None.
        partition (Union[Tuple[float,float,float], Dict[str, Tuple[float, float, float]]], optional): Partition of the dataset into train, validation, and test. Can be a tuple of three floats or a dictionary with 'train', 'val', and 'test' keys. Defaults to (0.8,0.1,0.1).

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing the train, validation, and test dataloaders.
    """
    # Get the dataset
    # NOTE: add configure total number of mols for each ds?
    for dataset in datasets:
        if isinstance(dataset, str):
            dataset = Path(dataset)
        if isinstance(dataset, Path):
            assert dataset.exists(), f"Dataset path {dataset} does not exist."
        elif not isinstance(dataset, Dataset):
            raise ValueError(f"Dataset must be a Path or Dataset, but got {type(dataset)}")
        
    dataset = Dataset()
    for ds in datasets:
        if isinstance(ds, Dataset):
            dataset += ds
        elif isinstance(ds, Path) or isinstance(ds, str):
            print(f"Loading dataset from {ds}...")
            dataset += Dataset.load(ds)
        else:
            raise ValueError(f"Unknown type for dataset: {type(ds)}")

    # Remove uncommon features for enabling batching
    dataset.remove_uncommon_features()

    # Get the split ids
    # load if path in config, otherwise generate. For now, always generate it.
    if splitpath is not None:
        raise NotImplementedError("split loading is not implemented yet.")
    else:
        split_ids = dataset.calc_split_ids(partition=partition)

    tr, vl, te = dataset.split(*split_ids.values())

    # Get the dataloaders
    train_loader = GraphDataLoader(tr, batch_size=train_batch_size, shuffle=True, num_workers=train_loader_workers, pin_memory=pin_memory, conf_strategy=conf_strategy)
    val_loader = GraphDataLoader(vl, batch_size=val_batch_size, shuffle=False, num_workers=val_loader_workers, pin_memory=pin_memory, conf_strategy=conf_strategy)
    test_loader = GraphDataLoader(te, batch_size=test_batch_size, shuffle=False, num_workers=test_loader_workers, pin_memory=pin_memory, conf_strategy=conf_strategy)

    return train_loader, val_loader, test_loader