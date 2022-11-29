import os

import torch
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from wandb_writer import WanDBWriter

from fastspeech2.configs import train_config
from fastspeech2.model.fastspeech2 import FastSpeech2
from fastspeech2.loss import FastSpeech2Loss
import dataset as ds
from tqdm import tqdm


model = FastSpeech2()
model = model.to('cuda')
fastspeech_loss = FastSpeech2Loss()
current_step = 0
buffer = ds.get_data_to_buffer(train_config)

dataset = ds.BufferDataset(buffer)

training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=ds.collate_fn_tensor,
    drop_last=True,
    num_workers=0
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

logger = WanDBWriter(train_config)

tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

for epoch in range(train_config.epochs):
    for i, batchs in enumerate(training_loader):
        # real batch start here
        for j, db in enumerate(batchs):

            current_step += 1
            tqdm_bar.update(1)

            logger.set_step(current_step)

            # Get Data
            character = db["text"].long().to('cuda')
            mel_target = db["mel_target"].float().to('cuda')
            max_src_pos = db['max_src_pos']
            duration = db["duration"].int().to('cuda')
            mel_pos = db["mel_pos"].long().to('cuda')
            src_pos = db["src_pos"].long().to('cuda')
            max_mel_len = db["mel_max_len"]
            target_energy = db['energy'].to('cuda')
            target_pitch = db['pitch'].to('cuda')
            # Forward
            mel_output, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len = model(character,
                                                                                                      src_pos,
                                                                                                      max_src_pos,
                                                                                                      mel_target,
                                                                                                      mel_pos,
                                                                                                      max_mel_len,
                                                                                                      duration,
                                                                                                      target_pitch,
                                                                                                      target_energy)
            # Calc Loss
            mel_loss, pitch_loss, energy_loss, duration_loss = fastspeech_loss(mel_output,
                                                                               mel_target,
                                                                               e_prediction,
                                                                               target_energy,
                                                                               d_prediction,
                                                                               duration,
                                                                               p_prediction,
                                                                               target_pitch
                                                                               )
            total_loss = mel_loss + pitch_loss + energy_loss + duration_loss

            # Logger
            t_l = total_loss.detach().cpu().numpy()
            m_l = mel_loss.detach().cpu().numpy()
            d_l = duration_loss.detach().cpu().numpy()
            e_l = energy_loss.detach().cpu().numpy()
            p_l = pitch_loss.detach().cpu().numpy()

            logger.add_scalar("duration_loss", d_l)
            logger.add_scalar("mel_loss", m_l)
            logger.add_scalar("total_loss", t_l)
            logger.add_scalar("energy_loss", e_l)
            logger.add_scalar("pitch_loss", p_l)

            # Backward
            total_loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip_thresh)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if current_step % train_config.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join('./model_saved', 'checkpoint_%d.pth.tar' % current_step))
                print("save model at step %d ..." % current_step)
