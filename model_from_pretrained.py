import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.bert.modeling_bert import BertModel

# partially based on/adapted from the Spec2Pep class in Casanovo (https://github.com/Noble-Lab/casanovo/blob/main/casanovo/denovo/model.py)
# using a pretrained NLP model (either BERT or GPT2) following "Pretrained Transformers as Universal Computation Engines" https://arxiv.org/pdf/2103.05247.pdf
class SpecBERTUniversalComputation(pl.LightningModule):
    def __init__(self,
                 vocab,
                 vocab_size, 
                 max_token_length, 
                 lr, 
                 weight_decay, 
                 warmup_iters, 
                 max_iters, 
                 model_name):
        super().__init__()
        self.save_hyperparameters()

        self.vocab = vocab

        if model_name == 'BERT':
            self.pretrained_model = BertModel.from_pretrained('bert-base-uncased')
        elif model_name == 'GPT2':
            self.pretrained_model = GPT2Model.from_pretrained('gpt2')
        else:
            raise ValueError("Only BERT and GPT2 are currently supported")
        
        # following the Lu paper, only fine-tune the layer norm and positional embeddings
        # (they are named differently in BERT and GPT2)
        for name, param in self.pretrained_model.named_parameters():
            if 'ln' in name or 'LayerNorm' in name or 'wpe' in name or 'position_embeddings' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  

        # note: both pretrained models have an embedding dimension of 768
        # They also expect the padding token to be at index 0
        self.input_layer = nn.Embedding(vocab_size, 768)
        self.output_layer = nn.Linear(768, vocab_size) # logits of each token

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.mz_token_loss = torch.nn.NLLLoss(reduction='mean', ignore_index=self.vocab.pad_token.token_index)

        self.intensity_layer = torch.nn.Linear(max_token_length, 768)

        self.warmup_iters = warmup_iters
        self.max_iters = max_iters

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, batch, batch_idx):
        masked_spectra = batch[0]

        mz_tokens = masked_spectra[:, :, 0].int()
        intensities = masked_spectra[:, :, 1:]

        embedding = self.input_layer(mz_tokens) + self.intensity_layer(intensities)

        # true for peaks that are not padding
        attention_mask = ~(masked_spectra == self.vocab.pad_token.token_index)[:,:,0]

        transformer_output = self.pretrained_model(inputs_embeds=embedding, encoder_attention_mask=attention_mask).last_hidden_state

        batch_size, sequence_length, embedding_size = transformer_output.shape

        mz_predictions = self.output_layer(transformer_output)

        return mz_predictions


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


        mz_tokens = masked_spectra[:, :, 0].int()
        intensities = masked_spectra[:, :, 1:]

        embedding = self.input_layer(mz_tokens) + self.intensity_layer(intensities)

        # true for peaks that are not padding
        attention_mask = ~(masked_spectra == self.vocab.pad_token.token_index)[:,:,0]

        transformer_output = self.pretrained_model(inputs_embeds=embedding, encoder_attention_mask=attention_mask).last_hidden_state

        batch_size, sequence_length, embedding_size = transformer_output.shape

        mz_predictions = self.output_layer(transformer_output)
        mz_softmaxes = self.softmax(mz_predictions)
        avg_mz_loss = self.mz_token_loss(mz_softmaxes.transpose(1, 2), masked_mz_labels.long())

        # TODO eventually add intensity prediction and loss
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
        # batch_output = saved_bert.forward(batch, 0)
        #mz_predictions = batch_output

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