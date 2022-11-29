import os

import text
from fastspeech2.audio import tools
from fastspeech2.configs import train_config
from fastspeech2.model.fastspeech2 import FastSpeech2
import torch
import numpy as np
from tqdm import tqdm

device = torch.device("cpu")


model = FastSpeech2()
checkpoint = torch.load('checkpoint_9195.pth.tar',  map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
print(model)


def synthesis(model, text, iter, alpha=1.0, beta=1.0, gamma=1.0):
    batch_text = []
    batch_src_pos = []

    for _ in range(16):
        batch_text.append(text)
        text_1 = np.array(text)
        text_1 = np.stack([text])
        src_pos = np.array([i + 1 for i in range(text_1.shape[1])]).tolist()
        batch_src_pos.append(src_pos)
        max_src_pos = max(src_pos)
    batch_text = torch.from_numpy(np.array(batch_text)).long()
    batch_src_pos = torch.from_numpy(np.array(batch_src_pos)).long()

    with torch.no_grad():
        mel, _, _, _ = model.forward(batch_text.to(device), batch_src_pos.to(device), max_src_pos,
                                     e_control=alpha, d_control=beta,p_control=gamma)
    return mel[iter].cpu().transpose(0, 1)


def get_data():
    tests = [
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space", ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)

    return data_list


data_list = get_data()
for energy in [0.8, 1., 1.3]:
    for i, phn in tqdm(enumerate(data_list)):
        mel = synthesis(model, phn, i, alpha=energy)

        os.makedirs("results", exist_ok=True)

        tools.inv_mel_spec(
            mel, f"results/s={energy}_{i}.wav"
        )

for i, phn in tqdm(enumerate(data_list)):
    mel = synthesis(model, phn, i, alpha=0.8, beta=0.8, gamma=0.8)

    os.makedirs("results", exist_ok=True)

    tools.inv_mel_spec(
        mel, f"results/s_all={0.8}_{i}.wav"
    )

for i, phn in tqdm(enumerate(data_list)):
    mel = synthesis(model, phn, i, alpha=1.2, beta=1.2, gamma=1.2)

    os.makedirs("results", exist_ok=True)

    tools.inv_mel_spec(
        mel, f"results/s_all={1.2}_{i}.wav"
    )


for i, phn in tqdm(enumerate(data_list)):
    mel = synthesis(model, phn, i, alpha=1, beta=1, gamma=1)

    os.makedirs("results", exist_ok=True)

    tools.inv_mel_spec(
        mel, f"results/base_configs={i}.wav"
    )
