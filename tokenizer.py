import torch
import sortedcollections
import math

from torch.nn.utils.rnn import pad_sequence

class SpectraTokenizer:
    def __init__(self, vocab, max_token_length, token_size):
        self.vocab = vocab
        self.max_token_length = max_token_length
        self.token_size = token_size # The granularity, provided as as a string. For example, for a bin size of 1.0, token_size = "1.0", while for a bin size of 0.2, token_size = "0.2"

    def tokenize_batch(self, spectra, intensity_mean, intensity_stdev):
        tokenized_spectra = []
        for i in range(0, len(spectra)):
            tokenized_spectrum = self.tokenize_spectrum(spectra[i], intensity_mean, intensity_stdev)
            tokenized_spectra.append(tokenized_spectrum)

        # pad with pad_token instead of zero since 0 is a real m/z token index
        padded_seq = pad_sequence(tokenized_spectra, batch_first=True, padding_value = self.vocab.pad_token.token_index)

        return padded_seq

    # Returns: tokenized spectrum, a Tensor of shape: (num_peaks, 1+max_token_length); each peak is the token index followed by all intensities in the peak
    def tokenize_spectrum(self, spectrum, intensity_mean, intensity_stdev):
        # first, get the unique (binned) peaks in the spectrum
        # If multiple mzs end up in the same token (bin), keep only the one with the highest intensity
        all_peaks_in_spectrum = sortedcollections.SortedDict()

        for peak in spectrum:
            discrete_mz = to_token(peak[0].item(), self.token_size)

            if discrete_mz in all_peaks_in_spectrum:
                previous_intensity = all_peaks_in_spectrum[discrete_mz]
            else:
                previous_intensity = -math.inf

            normalized_intensity = normalize_intensity(peak[1].item(), intensity_mean, intensity_stdev)
            all_peaks_in_spectrum[discrete_mz] = max(previous_intensity, normalized_intensity)

        # to be converted to a tensor representing the spectrum.
        # Each entry should be of the form (token_index, I1, In), where n = max_token_length
        # and each I represents the intensity of a peak in the token. For tokens smaller than 
        # max_token_length (e.g. a single m/z value), trailing I values are 0
        tokenized_peaks = []

        # Match peaks and remove them from the dictionary until every peak in the spectrum is accounted for
        while(len(all_peaks_in_spectrum) > 0):
            first_peak = all_peaks_in_spectrum.popitem(0)
            first_mz = first_peak[0]
            first_intensity = first_peak[1]

            # get all tokens from the vocabulary that contain this peak
            vocab_tokens_containing_peak = self.vocab.tokens_containing_peak(first_mz)

            # we already removed the first mz, but if we match a token we need to remove the subsequent peaks in the token as well
            extra_peaks_to_remove = []

            if len(vocab_tokens_containing_peak) == 0:
                # this m/z is not even in the vocabulary on its own, so use the UNK token
                # this should only happen when using non-exhaustive single m/z vocabularies
                peak_tensor = torch.zeros(1 + self.max_token_length)
                peak_tensor[0] = self.vocab.unk_token.token_index
                peak_tensor[1] = first_intensity
                tokenized_peaks.append(peak_tensor)

                continue

            # We know there is at least one token for this peak
            # Iterate over the possible tokens starting with the highest priority and use the first we find in the spectrum
            for possible_token in vocab_tokens_containing_peak:
                if possible_token.length == 1:
                    # The highest-priority token is just this mz; use it
                    peak_tensor = torch.zeros(1 + self.max_token_length) # store the mz token index and the zero-padded intensities
                    peak_tensor[0] = possible_token.token_index
                    peak_tensor[1] = first_intensity
                    tokenized_peaks.append(peak_tensor)
                    break
                else:
                    potential_mzs_to_remove = []
                    potential_intensities = [first_intensity]

                    # get the mzs of the possible token and see if all are left in the spectrum
                    mzs_to_match = possible_token.mzs
                    failed_mz_match_for_this_token = False
                    for possible_mz in mzs_to_match:
                        possible_mz = int(possible_mz)
                        if possible_mz == first_mz:
                            continue # We already know the "current" mz in in the spectrum
                        if possible_mz in all_peaks_in_spectrum:
                            potential_mzs_to_remove.append(possible_mz)
                            potential_intensities.append(all_peaks_in_spectrum[possible_mz])
                        else:
                            # can't match this token in the spectrum
                            failed_mz_match_for_this_token = True
                            break

                    if failed_mz_match_for_this_token:
                        continue # skip to the next token    

                    # By now we know the token is present in the spectrum
                    peak_tensor = torch.zeros(1 + self.max_token_length)
                    peak_tensor[0] = possible_token.token_index

                    for i in range(0, len(potential_intensities)):
                        peak_tensor[i + 1] = potential_intensities[i]

                    tokenized_peaks.append(peak_tensor)

                    extra_peaks_to_remove = potential_mzs_to_remove
                    break    

            # Remove the peaks that were part of a used token
            for mz_to_remove in extra_peaks_to_remove:
                del all_peaks_in_spectrum[mz_to_remove] 

        return torch.stack(tokenized_peaks)

# Token size is passed as a string to avoid floating-point arithmetic errors,
# as the output depends on the number of decimal places. It must be passed as a string with at least 
# one value after the decimal place (e.g. to have a token size of 1, str_token_size should be 1.0)
def to_token(mz_float, str_token_size):
    token_size = float(str_token_size)

    discretized_mz = round(mz_float / token_size) * token_size

    num_decimal_places = len(str_token_size.split(".")[1].rstrip("0"))

    return int(discretized_mz * pow(10, num_decimal_places))

# Normalize intensity to 0 mean and unit variance based on the mean and std of all intensities in the batch
def normalize_intensity(intensity, intensity_mean, intensity_std):
    return (intensity - intensity_mean) / intensity_std    
