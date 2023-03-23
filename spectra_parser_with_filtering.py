from depthcharge.data import parsers

import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict

# Extends the MgfParser from Depthcharge https://github.com/wfondrie/depthcharge
# to allow loading only a subset of spectra from the file, and limit the number of 
# spectra for each peptide
class MGFParserWithFiltering(parsers.MgfParser):
    def __init__(
        self,
        ms_data_file,
        start_spec_ix,
        end_spec_ix,
        max_spectra_per_peptide=1,
        ms_level=2,
        valid_charge=None,
        annotations=False
    ):
        """Initialize the MgfParser."""
        super().__init__(
            ms_data_file,
            ms_level=ms_level,
            valid_charge=valid_charge,
            annotations=annotations
        )    

        self.start_spec_ix = start_spec_ix
        self.end_spec_ix = end_spec_ix
        self.max_spectra_per_peptide = max_spectra_per_peptide

    def parse_spectrum(self, spectrum, count_per_peptide):
        """Parse a single spectrum.

        Parameters
        ----------
        spectrum : dict
            The dictionary defining the spectrum in MGF format.
        count_per_peptide : defaultdict
            The count of spectra read so far for each peptide    
        """
        if self.ms_level > 1:
            precursor_mz = float(spectrum["params"]["pepmass"][0])
            precursor_charge = int(spectrum["params"].get("charge", [0])[0])
        else:
            precursor_mz, precursor_charge = None, 0

        if self.annotations is not None:
            peptide = spectrum["params"].get("seq")
            self.annotations.append(peptide)
            count_per_peptide[peptide] += 1

        if self.valid_charge is None or precursor_charge in self.valid_charge:
            self.mz_arrays.append(spectrum["m/z array"])
            self.intensity_arrays.append(spectrum["intensity array"])
            self.precursor_mz.append(precursor_mz)
            self.precursor_charge.append(precursor_charge)
            self.scan_id.append(self._counter)

        self._counter += 1

    def read(self):
        """Read the ms data file"""

        current_nonduplicated_ix = 0
        count_per_peptide = defaultdict(lambda:0)

        with self.open() as spectra:
            for spectrum in tqdm(spectra, desc=str(self.path), unit="spectra"):
                if current_nonduplicated_ix > self.end_spec_ix:
                    break

                # skip spectra for peptides that already have the max spectra; these don't contribute to the start and end index
                if (self.annotations is not None) and (count_per_peptide[spectrum["params"].get("seq")] >= self.max_spectra_per_peptide):
                    continue

                # by now, we know we have a spectrum that isn't a duplicate peptide, so now the index can be used
                # e.g. if start_spec_ix is 1000, we skip all spectra before the 1000th spectrum that is not for a duplicate peptide
                if current_nonduplicated_ix < self.start_spec_ix:
                    current_nonduplicated_ix += 1
                    continue
                
                self.parse_spectrum(spectrum, count_per_peptide)
                current_nonduplicated_ix += 1

        self.precursor_mz = np.array(self.precursor_mz, dtype=np.float64)
        self.precursor_charge = np.array(
            self.precursor_charge,
            dtype=np.uint8,
        )

        self.scan_id = np.array(self.scan_id)

        # Build the index
        sizes = np.array([0] + [s.shape[0] for s in self.mz_arrays])
        self.offset = sizes[:-1].cumsum()
        self.mz_arrays = np.concatenate(self.mz_arrays).astype(np.float64)
        self.intensity_arrays = np.concatenate(self.intensity_arrays).astype(
            np.float32
        )