# audiojnd

Audio pair JND

Installation:
```
# lameenc is cooler but harder to use
sudo apt-get install -y lame libsox-fmt-all sox ffmpeg python3-pip
pip3 install -U tqdm sox click pydub audiomentations
pip3 install git+https://github.com/turian/torchopenl3.git
```

Download background noise and impulse response:
```
#[could also have colored noise]
wget https://github.com/karoldvl/ESC-50/archive/master.zip
mkdir -p data/esc-50/ && pushd data/esc-50/ && unzip ../../master.zip && rm ../../master.zip && popd
# Could also get echothief: http://www.echothief.com/
wget https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip
mkdir -p data/MIT-McDermott-ImpulseResponse/ && pushd data/MIT-McDermott-ImpulseResponse/ && unzip ../../Audio.zip && rm ../../Audio.zip && popd
```

You might need LLVM:
```
wget --no-check-certificate -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository 'deb http://apt.llvm.org/bionic/   llvm-toolchain-bionic-10  main'
sudo apt update
sudo apt install llvm
sudo apt-get install llvm-10 lldb-10 llvm-10-dev libllvm10 llvm-10-runtime
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-6.0 6
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-10 10
sudo update-alternatives --config llvm-config
LLVM_CONFIG=/usr/bin/llvm-config-10 CXXFLAGS=-fPIC pip3 install llvmlite
```

Usage:
```
./get_fsd50.py
./get_backgroundnoise.py
./preprocess.py
./transforms.py
```
