import model_dataloader
import masker
import tokenizer
import vocab
import tempfile
import uuid
import numpy as np
import os
import model
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from hdf5_with_filtering import AnnotatedSpectrumIndex
from glob import glob

vocab = vocab.BasicSpectraVocab(1)    
vocab.load_from_file("/disk/dragon-storage/homes/seber007/single-tokens-mar22.txt") # todo save vocab to that file
tokenizer1 = tokenizer.SpectraTokenizer(vocab, 1, "1.0")
masker1 = masker.Masker(vocab, 0.15, 0.8, 0.1)

train_filenames = glob(os.path.join('/disk/dragon-storage/homes/seber007/data/spectra/massive-kb/v2_hcd_only', '*'))

tmp_dir = tempfile.TemporaryDirectory()
max_charge=10
train_idx_fn = os.path.join(tmp_dir.name, f"{uuid.uuid4().hex}.hdf5")
train_filenames = glob(os.path.join('/disk/dragon-storage/homes/seber007/data/spectra/massive-kb/v2_hcd_only', '*'))
valid_charge = np.arange(1, max_charge + 1)
train_index = AnnotatedSpectrumIndex(
    train_idx_fn, start_spec_ix = 200_000, end_spec_ix=600_000, ms_data_files=train_filenames, valid_charge=valid_charge
)

dataloader_params = dict(
        batch_size=32,
        n_peaks=150,
        min_mz=50.0,
        max_mz=2500.0,
        min_intensity=0.01,
        remove_precursor_tol=2.0,
        vocabulary=vocab,
        tokenizer=tokenizer1,
        masker=masker1,
        random_seed=23,
        n_workers=40
)

train_loader = model_dataloader.SpecBertDataModule(
    data_index=train_index, **dataloader_params
)
train_loader.setup()

train_data_loader = train_loader.train_dataloader()
val_data_loader = train_loader.val_dataloader()

# set up callback to save model:
callbacks = [
    pl.callbacks.ModelCheckpoint(
    save_top_k=-1,
    save_weights_only=True,
    every_n_train_steps=1000,
    )
]

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

model1 = model.SpecBERT(vocab=vocab, vocab_size=len(vocab.all_tokens), max_token_length = 1, embed_dimension=512, hidden_dimension=1024, n_attn_layers=12, n_attn_heads=8, lr=1e-4, weight_decay=1e-5, warmup_iters=100_000, max_iters=600_000)

seed_everything(42, workers=True)

ts_logger = pl_loggers.TensorBoardLogger(save_dir="test-march27-GPT2/")

trainer = pl.Trainer(
        precision=16,
        accelerator="auto",
        devices="auto",
        auto_select_gpus=True,
        callbacks=callbacks,
        check_val_every_n_epoch=1, 
        max_epochs=1000,
        log_every_n_steps=50,
        logger=ts_logger
)

trainer.fit(
    model1, train_data_loader, val_data_loader
)

tmp_dir.cleanup()