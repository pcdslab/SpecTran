import torch
import random
import numpy.random as np_rand

class Masker:

    def __init__(self, vocab, probability, mask_probability, random_probability):
        '''
            change_token_chance: overall percent of the words in the spectrum to mask (e.g. 15% bert)
            mask_probability: of those 15% of words, how many should become the mask token? 
            random_probability: of those 15%, how many random character
            stay_same_probability is implied by the sum of the rest
        '''
        self.vocab = vocab
        self.probability = probability
        self.mask_probability = mask_probability
        self.random_probability = random_probability

    # spectra: torch.tensor of shape (batch_size, peak_count, 1 + max_token_length)
    # the third dimension consists of the token index, and the corresponding intensity values for each peak in the token
    def mask_batch(self, spectra):
        all_masked_specs = []
        all_spec_labels = []

        for spec in spectra:
            masked_spec, labels = self.mask_single(spec)

            all_masked_specs.append(masked_spec)
            all_spec_labels.append(labels)

        return torch.stack(all_masked_specs), torch.nn.utils.rnn.pad_sequence(all_spec_labels, batch_first=True, padding_value = self.vocab.pad_token.token_index)

    # spec: torch.tensor of shape (peak_count, 1 + max_token_length)
    def mask_single(self, spec): 
        # find the indices of non-padding token values
        mz_col = spec[:, 0].int()
        real_token_indices = torch.nonzero(mz_col != self.vocab.pad_token.token_index)   # tensor of shape (non-padding-token-count, 1), containing every index in spec that is not padding

        # make sure we mask at least one peak
        num_peaks_to_mask = max(1, int(self.probability * real_token_indices.size(0)))

        # select the indices to mask in the spectrum from the non-padding indices
        indices_to_mask = torch.randperm(real_token_indices.size(0))[:num_peaks_to_mask]
        indices_to_mask = indices_to_mask.view(len(indices_to_mask), 1).long()

        labels = []

        for index_of_peak_to_mask in indices_to_mask:
            index = index_of_peak_to_mask.item()

            original_peak = spec[index]
            masked_peak_label = torch.cat((index_of_peak_to_mask, torch.clone(original_peak)))
            labels.append(masked_peak_label)

            random_value = random.random()
            if random_value > 1.0 - self.random_probability:
                # replace with a random token

                # the number of real peaks in the token to mask - we only replace a token with a token of the same size
                original_token_length = len(torch.nonzero(original_peak[1:]))

                # get a random token of the same length as the original                
                replacement_token = self.vocab.get_random_token(original_token_length)
                spec[index][0] = replacement_token.token_index

                # replace every non-padding intensity value with a random sample from N(0, 1)
                for k in range(1, original_token_length + 1):
                    spec[index][k] = np_rand.default_rng().normal(0, 1)
            elif random_value < self.mask_probability:
                # replace with the mask token
                original_token_length = len(torch.nonzero(original_peak[1:]))

                replacement_token = self.vocab.get_mask_token()

                spec[index][0] = replacement_token.token_index

                # replace every non-padding intensity value with a random sample from N(0, 1)
                for k in range(1, original_token_length + 1):
                    spec[index][k] = np_rand.default_rng().normal(0, 1)
            else:
                # keep the token unchanged
                continue     

        return spec, torch.stack(labels)