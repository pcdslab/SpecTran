# adapted from Casanovo's dataloader class https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/dataloaders.py
# updated to use only one data file and split into training and validation sets, instead of passing the train and validation files separately
import functools
import os
from typing import List, Optional, Tuple
import torch
from torch.utils.data import random_split
import numpy as np
import pytorch_lightning as pl

from casanovo.data.datasets import AnnotatedSpectrumDataset
from hdf5_with_filtering import AnnotatedSpectrumIndex

class SpecBertDataModule(pl.LightningDataModule):
    def __init__(
        self,
        vocabulary,
        tokenizer, 
        masker,
        data_index: AnnotatedSpectrumIndex,
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
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.masker = masker
        self.train_ratio = train_ratio

    def setup(self) -> None:
        make_dataset = functools.partial(
            AnnotatedSpectrumDataset,
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
            collate_fn=functools.partial(prepare_batch, tokenizer=self.tokenizer, masker=self.masker),
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
    batch: List[Tuple[torch.Tensor, float, int, str]],
    tokenizer,
    masker
) -> Tuple[torch.Tensor, torch.Tensor]:

    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))

    # calculate the mean and std of all intensities in the batch, to use for normalization in the tokenizer
    all_intensities_in_batch = []
    for spectrum in spectra:
        for peak in spectrum:
            intensity = peak[1]
            all_intensities_in_batch.append(intensity)
    intensity_stdev = np.std(all_intensities_in_batch)
    intensity_mean = np.mean(all_intensities_in_batch)

    tokenized_spectra = tokenizer.tokenize_batch(spectra, intensity_mean, intensity_stdev)

    # tokenized spectra shape: torch.Size([batch_size, sequence_length, 3]) (batch_size spectra, sequence_length peaks, [tokenIx, I1, I2]) 
    # as many I values as max token length; unused I values are "padded" with 0s
    # the overall sequence is padded to sequence_length using vocab.pad_token (padding with 0 is invalid since 0 is a real mz token index)
    
    # labels: each spectra has a list of labels (mz-intensity "groups", all must be predicted)
    masked_tokenized_spectra, mask_labels = masker.mask_batch(tokenized_spectra)

    return masked_tokenized_spectra, mask_labels