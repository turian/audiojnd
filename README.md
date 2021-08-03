# audiojnd

Audio pair JND

Installation:
```
# lameenc is cooler but harder to use
sudo apt-get install -y lame libsox-fmt-all sox ffmpeg python3-pip
pip3 install -U tqdm sox click
pip3 install git+https://github.com/turian/torchopenl3.git
```

Usage:
```
./get_fsd50.py
./preprocess.py
./transforms.py
```
