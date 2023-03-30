## SpecTran
Repository for peptide and mass spectrometry MS/MS data language models.

This repository provides functionality to pretrain a BERT-style masked language model to predict missing peaks in 
tandem mass spectra using a transformer encoder. It is possible to train the transformer from scratch, or to 
use a Frozen Pretrained Transformer (FPT) model (BERT or GPT2) according to "Pretrained Transformers as Universal Computation Engines"
(https://arxiv.org/abs/2103.05247) and fine-tune only the input, output and layer norm parameters.

## Dependencies
* Numpy
* Pytorch
* Pytorch lightning
* Depthcharge (https://github.com/wfondrie/depthcharge)

## Vocabulary / Tokenization

Before training the model, it is necessary to initialize the token vocabulary of m/z values. This can be done using the generate_vocab.ipynb script. The script allows you to choose the granularity/bin width of the tokens by providing a "token_size" string.

For example, for a bin witdth of 1.0 m/z, token_size should be "1.0", while for a bin width of 0.1, token_size should be "0.1"

The script calculates the frequency of all "single" tokens (individual m/z values), as well as the top pair_token_count pair tokens (calculated by considering all possible combinations of peaks in the training data). The output file will be sorted from most frequent to least frequent, in the following format:

Single tokens: (discretized_mz, frequency)
Pair tokens: ((discretized_mz_1, discretized_mz_2), frequency)

Note that the frequency of a single token is defined as its frequency only when occurring on its own, not when it is part of one of the top pair_token_count pair tokens.

# Training the model

## Data

The model can be trained on any mgf file/files containing spectra. It is not necessary to split the input file into
train and validation, as that will be performed by the dataloader itself.

## Training

The training process is illustrated in the scripts training_script_example_fpt_singles.py (for the model using a frozen pretrained BERT)
and training_script_example_fromscratch_singles.py (for the model training from scratch). 

## Initializing the vocabulary, tokenizer and masker 
To train a model, first you must load the vocabulary from a saved file as discussed in the section above. 

The vocabulary contains all tokens from the file, indexed starting from the lowest, as well as the special tokens pad, mask, and unk. 
By convention, the <pad> token is at index 0, the <mask> token is at index 1, and the <unk> token is at index 2. Indexes 3..vocabulary_size+2
contain the "real" tokens, ordered most frequent first.

Then, create a tokenizer and masker with the appropriate parameters:

* max_token_length: the maximum number of m/z peaks in a token. Setting this to 1 treats each "single" peak as a token, while setting this to 2 also uses pairs of token. Note that this must be set according to the saved vocabulary you are using

* token_size: a string indicating the desired granularity of the tokens. "1.0" means use a token "bin" size of 1.0, "0.1" means use a "bin" size of 0.1, etc. If multiple m/z peaks fall into the smae "bin", only the one with the highest intensity will be kept. Note that this must be consistent with the saved vocabulary. 

* probability: the probability that a token is selected for maksing (recommended: 15%)

* mask_probability: of the tokens that are selected for masking, this percent will be replaced with the mask token (recommended: )

* random_probability: of the tokens that are selected for masking, this percent will be replaced with a random token (the remainder after summing this and mask_probability will be kept as the original token, but still considered "masked" for the purposes of prediction)

## Creating the dataloader

The data is loaded with modified version of several utilities provided by Casanovo (https://github.com/Noble-Lab/casanovo).

The data loading process works by creating an hdf5 file in a temporary directory using the AnnotatedSpectrumindex class (originally from Casanovo; modified in this repository). This class takes the following parameters:

* max_charge: only allow spectra with at most this charge level
* max_spectra_per_peptide: Only allow this many spectra for each peptide label. Any duplicate spectra after this count are ignored (not loaded)
* start_spec_ix: the starting index (in the original mgf file) of the first spectrum to include. Note that this refers only to spectra that are not ignored by the max_spectra_per_peptide parameter (e.g. if start_spec_ix is 3 and max_spectra_per_peptide is 1, the dataloader will start with the third *unique* spectrum, which might not actually be the third in the file if there are duplicates before it.
* end_spec_ix: the index of the last spectra to load, following the same logic as start_spec_ix

The model then creates a dataloader (SpecBertDataModule), which processes the raw spectra into their tokenized and masked versions and returns the output in batches of tensors. 

Each batch is formatted as a tuple(masked_tokenized_spectra, mask_labels)

masked_tokenized_spectra is a tensor of shape (batch_size, peak_count, 1 + max_token_length). Each peak's 0th value is the m/z token index in the vocabulary, and the subsequent values are the associated intensity values (there are only multiple intensity values if the token has multiple m/z values; otherwise they are padded with zeros). The tensor is padded to the length of the longest spectrum. 

mask_labels is a tensor of shape (batch_size, number of masked peaks, 2 + max_token_length)
    - The 0th value is the index in the spectrum of the peak that was masked
    - The next value is the index of the original m/z token
    - Subsequent values are the intensities

Then, the dataloader is created from the index with the following parameters:

* n_peaks: the number of peaks to include in each spectrum, keeping those with the highest intensity
* min_mz: filter peaks with m/z below this value
* max_mz: filter peaks with m/z above this value
* min_intensity: filter peaks with intensity below this value
* remove_precursor_tol: peaks within this distance of the precursor mass are removed

Note that in the current setup, the data loading utilities expect labeled spectra. However, outside of the filtering based on charge and unique peptides, the labels are not actually used. Therefore, these classes will soon be modified to also allow using unlabeled spectra.

## Training the model

After preparing the dataloader, the model is trained using Pytorch Lightning's Trainer.

Below are relevant model parameters:

* max_token_length: the maximum number of m/z values for token; must be consistent with the vocabulary
* model_name (if using a FPT pretrained model instead of training from scratch): either "BERT" or "GPT2"

If training from scratch:
* embed_dimension
* hidden_dimension
* n_attn_layers
* n_attn_heads

## Logs

The training script is configured to log to tensorboard by default, to the directory set in the script. To view the tensorboard logs on your local computer, you can ssh to the cluster as follows (assuming tensorboard is running on port 6007 on the remote server):

ssh -L 16006:127.0.0.1:6007 {username}@dragon.cs.fiu.edu

This will allow you to view tensorboard on your local computer by accessing localhost:16006/ in a browser
