import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn

# partially based on/adapted from the Spec2Pep class in Casanovo (https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/model.py)
class SpecBERT(pl.LightningModule):
    def __init__(self,
                 vocab,
                 vocab_size, 
                 max_token_length, 
                 embed_dimension, 
                 hidden_dimension, 
                 n_attn_layers, 
                 n_attn_heads, 
                 lr, 
                 weight_decay, 
                 warmup_iters, 
                 max_iters):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab

        self.encoder = SpectrumEncoder(self.vocab, vocab_size, embed_dimension, hidden_dimension, n_attn_layers, n_attn_heads, max_token_length)
        
        self.regular_softmax = torch.nn.LogSoftmax(dim=-1)
        self.mz_decoder = torch.nn.Linear(embed_dimension, vocab_size) # convert each peak embedding to a value representing the probability of each token
        self.mz_token_loss = torch.nn.NLLLoss(reduction='mean', ignore_index=self.vocab.pad_token.token_index)

        # TODO calculate percentage correct

        # TODO eventually add an intensity prediction and loss
        #self.intensity_decoder = torch.nn.Linear(embed_dimension, max_token_length)
        #self.intensity_loss = torch.nn.MSELoss(reduction='mean')

        self.warmup_iters = warmup_iters
        self.max_iters = max_iters

        self.lr = lr
        self.weight_decay = weight_decay

    # takes a batch of (masked) and tokenized spectra and returns the embedding along with the predicted token for each peak
    def forward(self, batch, batch_idx):
        spectra = batch[0]

        spectra_embeddings = self.encoder(spectra)

        batch_size, sequence_length, embedding_size = spectra_embeddings.shape

        mz_predictions = self.mz_decoder(spectra_embeddings)
        mz_softmaxes = self.regular_softmax(mz_predictions)

        return mz_predictions, mz_softmaxes

    def training_step(self, batch, batch_idx, is_train=True):
        masked_spectra = batch[0] # 32, 108, 3
        mask_labels = batch[1] # 32, 16, 4

        # Create a labels tensor full of zeros for unmasked values and the real labels for masked values
        masked_mz_labels = torch.full_like(masked_spectra[:, :, 0].int(), self.vocab.pad_token.token_index)
        for spec_ix in range(0, len(masked_mz_labels)):
            mask_labels_for_this_spec = mask_labels[spec_ix] # shape: (number_of_masked_peaks, 4)
            # mask_labels_for_this_spec[i] is the label for the ith peak that was masked in this spectrum, in the form (index in spec, mz token index, I1, I2)

            for k in range(0, len(mask_labels_for_this_spec)): # for each masked peak in this spectrum
                peak_label = mask_labels_for_this_spec[k] # 4d tuple containing index in spectrum, index of mz token, I1, I2

                index_in_spectrum = peak_label[0].int()

                if index_in_spectrum == self.vocab.pad_token.token_index and k > 0:
                    # ignore padding tokens
                    continue

                mz_token_label = peak_label[1].int()

                # TODO also include intensity labels if/when we start to predict intensity
                # i_1 = peak_label[2]
                # i_2 = peak_label[3]

                masked_mz_labels[spec_ix][index_in_spectrum] = mz_token_label

        spectra_embeddings = self.encoder(masked_spectra)

        batch_size, sequence_length, embedding_size = spectra_embeddings.shape

        mz_predictions = self.mz_decoder(spectra_embeddings)
        mz_softmaxes = self.regular_softmax(mz_predictions)
        avg_mz_loss = self.mz_token_loss(mz_softmaxes.transpose(1, 2), masked_mz_labels.long())

        # intensity_predictions = self.intensity_decoder(spectra_embeddings) # torch.Size([32, 108, 2])
        # intensity_loss_result = self.intensity_loss(intensity_predictions, intensity_labels) # torch.Size([32, 108, 2]) (batch size, peak count, token length)
        
        if is_train:
            percent_correct = self.calculate_percent_correct(mz_predictions, mask_labels)

            self.log("Train loss", avg_mz_loss,  batch_size=batch_size, on_epoch=True, sync_dist=True)
            self.log("Train correct %", percent_correct,  batch_size=batch_size, on_epoch=True, sync_dist=True)
        else:    
            # validation logging
            percent_correct = self.calculate_percent_correct(mz_predictions, mask_labels)

            self.log("Val loss", avg_mz_loss,  batch_size=batch_size, on_epoch=True, sync_dist=True)
            self.log("Val correct %", percent_correct,  batch_size=batch_size, on_epoch=True, sync_dist=True)

        return avg_mz_loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, False)

        return loss

    # from Casanovo (https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/model.py)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.warmup_iters, max_iters=self.max_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}
    
    def calculate_percent_correct(self, mz_predictions, mask_labels):
        predictions = 0
        correct = 0

        mz_predictions_token = torch.max(mz_predictions, dim=2)

        token_predictions = mz_predictions_token[1]

        masked_indices = mask_labels[:, :, 0].int()
        mz_values = mask_labels[:, :, 1]

        for spec_ix in range(len(mask_labels)):        
            masked_peak_indices = masked_indices[spec_ix]
            masked_mz_labels = mz_values[spec_ix]

            for i in range(0, len(masked_peak_indices)):
                index_in_spectrum = masked_peak_indices[i]

                if index_in_spectrum == self.vocab.pad_token.token_index and i > 0: # TODO techincally this should be vocab.pad_token.token_index
                    break # non-leading zeros are padding

                real_mz_value = masked_mz_labels[i]

                predicted_mz_value = token_predictions[spec_ix][index_in_spectrum]

                predictions += 1

                if real_mz_value == predicted_mz_value:
                    correct += 1

        return correct / predictions


# from Casanovo (https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/model.py)
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, optimizer, warmup, max_iters
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch / self.warmup
        return lr_factor    

class SpectrumEncoder(torch.nn.Module):
    def __init__(self, vocab, vocab_size, embed_dimension, hidden_dimension, num_attn_layers, num_attn_heads, max_token_length):
        super(SpectrumEncoder, self).__init__()

        self.vocab = vocab
        self.vocab_size = vocab_size
        self.embed_dim = embed_dimension
        self.hidden_dim = hidden_dimension
        self.attn_layers = num_attn_layers
        self.attn_heads = num_attn_heads

        self.token_embedder = torch.nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.vocab.pad_token.token_index)

        # TODO consider other ways of representing intensity, since we only use all of max_token_length for tokens of the max length, and there are trailing zeros in the input if not
        self.intensity_layer = torch.nn.Linear(max_token_length, self.embed_dim)

        attentionLayer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dimension,
            nhead=num_attn_heads,
            dim_feedforward=hidden_dimension,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(attentionLayer, num_layers=num_attn_layers)

    def forward(self, spectra):
        # Spectra: a Tensor of dimension (batch_size, n_peaks, 1 + max_token_length)

        masked = (spectra == self.vocab.pad_token.token_index)[:,:,0] # torch.Size([batch_size, sequence_length]), true for peaks that are masked in spec

        mz_tokens = spectra[:, :, 0].int() # torch.Size([batch_size, sequence_length]); the mz token index (in the vocabulary) of each peak

        intensities = spectra[:, :, 1:] # torch.Size([batch_size, sequence_length, max_token_length]); all intensities in each peak

        embedding = self.token_embedder(mz_tokens) + self.intensity_layer(intensities)

        transformer_input = embedding.permute(1, 0, 2) # the transformer encoder expects the batch to be the second dimension

        return self.transformer_encoder(transformer_input, src_key_padding_mask=masked).permute(1, 0, 2)


# # Taken from https://github.com/taufique74/AdaptiveIO; modified to include padding index in the embeddings
# class AdaptiveInput(nn.Module):
#     def __init__(self, in_features, n_classes, cutoffs, padding_idx, div_value=4.0, head_bias=False, tail_drop=0.5):
#         super(AdaptiveInput, self).__init__()
#         cutoffs = list(cutoffs)

#         if (cutoffs != sorted(cutoffs)) \
#                 or (min(cutoffs) <= 0) \
#                 or (max(cutoffs) >= (n_classes - 1)) \
#                 or (len(set(cutoffs)) != len(cutoffs)) \
#                 or any([int(c) != c for c in cutoffs]):
#             raise ValueError("cutoffs should be a sequence of unique, positive "
#                              "integers sorted in an increasing order, where "
#                              "each value is between 1 and n_classes-1")

#         self.in_features = in_features
#         self.n_classes = n_classes
#         self.cutoffs = cutoffs + [n_classes]
#         self.div_value = div_value
#         self.head_bias = head_bias
#         self.tail_drop = tail_drop

#         self.n_clusters = len(self.cutoffs) - 1
#         self.head_size = self.cutoffs[0]

# #         self.head = nn.Sequential(nn.Embedding(self.head_size, self.in_features),
# #                                   nn.Linear(self.in_features, self.in_features, bias=self.head_bias))
        
#         self.head = nn.Embedding(self.head_size, self.in_features, padding_idx=padding_idx)
# #                                   nn.Linear(self.in_features, self.in_features, bias=self.head_bias))
        
#         self.tail = nn.ModuleList()

#         for i in range(self.n_clusters):
#             hsz = int(self.in_features // (self.div_value ** (i + 1)))
#             osz = self.cutoffs[i + 1] - self.cutoffs[i]

#             projection = nn.Sequential(
#                 nn.Embedding(osz, hsz, padding_idx=padding_idx),
#                 nn.Linear(hsz, self.in_features, bias=False),
#                 nn.Dropout(self.tail_drop)
#             )

#             self.tail.append(projection)

#     def forward(self, input):
#         used_rows = 0
#         input_size = list(input.size())

#         output = input.new_zeros([input.size(0) * input.size(1)] + [self.in_features]).half()
#         input = input.view(-1)

#         cutoff_values = [0] + self.cutoffs
#         for i in range(len(cutoff_values) - 1):

#             low_idx = cutoff_values[i]
#             high_idx = cutoff_values[i + 1]

#             input_mask = (input >= low_idx) & (input < high_idx)
#             row_indices = input_mask.nonzero().squeeze()

#             if row_indices.numel() == 0:
#                 continue
#             out = self.head(input[input_mask] - low_idx) if i == 0 else self.tail[i - 1](input[input_mask] - low_idx)
#             output.index_copy_(0, row_indices, out.half())
#             used_rows += row_indices.numel()

#         # if used_rows != input_size[0] * input_size[1]:
#         #     raise RuntimeError("Target values should be in [0, {}], "
#         #                        "but values in range [{}, {}] "
#         #                        "were found. ".format(self.n_classes - 1,
#         #                                              input.min().item(),
#         #                                              input.max().item()))
#         return output.view(input_size[0], input_size[1], -1)