# adapted from Casanovo's dataloader class https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/dataloaders.py
# updated to use only one data file and split into training and validation sets, instead of passing the train and validation files separately
import functools
import os
from typing import List, Optional, Tuple
import torch
from torch.utils.data import random_split
import numpy as np
import pytorch_lightning as pl

from casanovo.data.datasets import SpectrumDataset
from hdf5_with_filtering import SpectrumIndex

class TokenizerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_index: SpectrumIndex,
        batch_size: int = 128,
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: Optional[int] = None,
        random_seed: Optional[int] = None,
        train_ratio = 0.8
    ):
        super().__init__()
        self.data_index = data_index
        self.batch_size = batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.train_dataset = None
        self.valid_dataset = None
        self.train_ratio = train_ratio

    def setup(self) -> None:
        make_dataset = functools.partial(
            SpectrumDataset,
            n_peaks=self.n_peaks,
            min_mz=self.min_mz,
            max_mz=self.max_mz,
            min_intensity=self.min_intensity,
            remove_precursor_tol=self.remove_precursor_tol,
        )
        dataset = make_dataset(self.data_index, random_state=self.rng)
        gen = torch.Generator().manual_seed(self.random_seed)
        self.train_dataset, self.valid_dataset = random_split(dataset, [self.train_ratio, 1 - self.train_ratio], generator=gen)

    def _make_loader(
        self, dataset: torch.utils.data.Dataset
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader.
        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            A PyTorch Dataset.
        Returns
        -------
        torch.utils.data.DataLoader
            A PyTorch DataLoader.
        """
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=prepare_batch,
            pin_memory=True,
            num_workers=self.n_workers,
        )    

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(self.train_dataset)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset)  

    
def prepare_batch(
    batch: List[Tuple[torch.Tensor, float, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Collate MS/MS spectra into a batch.
    The MS/MS spectra will be padded so that they fit nicely as a tensor.
    However, the padded elements are ignored during the subsequent steps.
    Parameters
    ----------
    batch : List[Tuple[torch.Tensor, float, int, str]]
        A batch of data from an AnnotatedSpectrumDataset, consisting of for each
        spectrum (i) a tensor with the m/z and intensity peak values, (ii), the
        precursor m/z, (iii) the precursor charge, (iv) the spectrum identifier.
    Returns
    -------
    spectra : torch.Tensor of shape (batch_size, n_peaks, 2)
        The padded mass spectra tensor with the m/z and intensity peak values
        for each spectrum.
    precursors : torch.Tensor of shape (batch_size, 3)
        A tensor with the precursor neutral mass, precursor charge, and
        precursor m/z.
    spectrum_ids : np.ndarray
        The spectrum identifiers (during de novo sequencing) or peptide
        sequences (during training).
    """
    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    return spectra, precursors, np.asarray(spectrum_ids)