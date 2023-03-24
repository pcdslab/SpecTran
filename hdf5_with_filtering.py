from spectra_parser_with_filtering import MGFParserWithFiltering
from depthcharge.data import hdf5

# Adapted from the hdf5 class from Depthcharge https://github.com/wfondrie/depthcharge
# to allow loading only a subset of spectra from the file, and limit the number of 
# spectra for each peptide

class AnnotatedSpectrumIndex(hdf5.AnnotatedSpectrumIndex):
    def __init__(
        self,
        index_path,
        start_spec_ix, # the starting index of the spectra to return, considering only spectra meeting the max_spectra_per_peptide constraint
        end_spec_ix, # the ending index of the spectra to return, considering only spectra meeting the max_spectra_per_peptide constraint
        max_spectra_per_peptide=1,
        ms_data_files=None,
        ms_level=2,
        valid_charge=None,
        overwrite=False,
    ):
        self.start_spec_ix = start_spec_ix
        self.end_spec_ix = end_spec_ix
        self.max_spectra_per_peptide = max_spectra_per_peptide

        super().__init__(
            index_path,
            ms_data_files,
            ms_level,
            valid_charge,
            overwrite
        )

    def _get_parser(self, ms_data_file):
        """Get the parser for the MS data file"""
        if ms_data_file.suffix.lower() == ".mgf":
            return MGFParserWithFiltering(
                ms_data_file,
                start_spec_ix = self.start_spec_ix,
                end_spec_ix = self.end_spec_ix,
                max_spectra_per_peptide = self.max_spectra_per_peptide,
                ms_level=self.ms_level,
                valid_charge=self._valid_charge,
                annotations=True
            )

        raise ValueError("Only MGF files are currently supported.")
