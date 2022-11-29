## Installation guide
```
git clone https://github.com/leksious/TTS.git
wget https://www.dropbox.com/s/8up1qsdwn4pqnwo/checkpoint_9195.pth.tar  -P fastspeech2/
python fastspeech2/synthesize.py

В папке results появятся файлы wav, где первое число означает изменение энергии, а второе число номер озвучиваемой фразы
