# audiojnd

Audio pair JND

Installation:
```
pip3 install -U tqdm sox audiomentations
# lameenc is cooler but harder to use
sudo apt-get install -y lame libsox-fmt-all sox ffmpeg
```

Usage:
```
./get_fsd50.py
./get_backgroundnoise.py
./preprocess.py
./transforms.py
```
