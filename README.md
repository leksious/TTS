## Installation guide
```
git clone https://github.com/leksious/TTS.git
pip install -r requirements.txt
wget https://www.dropbox.com/s/8up1qsdwn4pqnwo/checkpoint_9195.pth.tar  -P fastspeech2/

UPD 30.11.22, 02.16: 
Для waveglow: 
скачать файл по ссылке https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view
закинуть его в папку fastspeech2/waveglow/pretrained_model

python fastspeech2/synthesize.py (лучше запустить ручками из пайчарма)


В папке results появятся файлы wav, где первое число означает изменение энергии, а второе число номер озвучиваемой фразы
