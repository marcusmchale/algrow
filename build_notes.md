## Initial setup of virtual environment (recommended)
### Linux
```
apt install python3.10
apt install python3.10-venv
python3 -m venv venv
. ./venv/bin/activate
```
### OSx
Install home brew and set path
```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
at the bottom of the output from that are two lines to set up the path environment
then install python 3.10 and libomp
```
brew install python3.10
brew install libomp
```
```
python3.10 -m venv venv
. ./venv/bin/activate
```

### Windows
install python 3.10 (through the app store works)
then in a normal windows cmd, activate the venv
```
python -m venv venv
venv\Scripts\activate  # if in powershell use activate.ps1 
```

## Build
### Wheel
Download
```git clone https://github.com/marcusmchale/algrow```
Install python packages
```python3.10-distutils python3.10-dev```
Build
```python setup.py bdist_wheel```

### Binary (PyInstaller)
Make sure to include licenses for all dependencies if packaging a binary for distribution.
#### On linux
Make sure pyinstaller and AlGrow requirements are installed
```
pip install pyinstaller
pip install -r REQUIREMENTS.txt
```
Install the following to the system
```
sudo apt install libspatialindex-dev
```
Then run pyinstaller in the algrow root path
You might want to check the path of libspatialindex files
```
pyinstaller --onefile --paths src/ --clean --noconfirm --log-level WARN \
--name algrow_0_6_3_linux \
--add-data=bmp/logo.png:./bmp/ \
--add-data=venv/lib/python3.10/site-packages/open3d/libc++*.so.1:. \
--add-data=venv/lib/python3.10/site-packages/Rtree.libs/libspatialindex-91fc2909.so.6.1.1:. \
--add-data=venv/lib/python3.10/site-packages/open3d/resources:./open3d/resources \
--add-data=/lib/x86_64-linux-gnu/libspatialindex*:. \
--hidden-import='PIL._tkinter_finder' \
algrow.py
```
#### On macosx
Make sure pyinstaller and AlGrow requirements are installed
```
pip install pyinstaller
pip install -r REQUIREMENTS.txt
```
```
pyinstaller --onefile --paths src/ --clean --noconfirm --log-level WARN \
--name algrow_0_5_0_osx \
--icon=./bmp/icon.ico \
--add-data=bmp/logo.png:./bmp/ \
--add-data=venv/lib/python3.10/site-packages/open3d/resources:./open3d/resources \
--hidden-import='PIL._tkinter_finder' \
algrow.py
``` 
#### On windows
- to be assessed on a fresh system
  - might need to install MS visual c++ redistributable, might be fine if install msvc-runtime before installing open3d
    - description: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
    - the exe: https://aka.ms/vs/17/release/vc_redist.x86.exe

In admin powershell (or cmd prompt) run the below to allow script execution,
This is needed to activate the venv, I didn't test if it is required subsequently as I left it activated.

```set-executionpolicy RemoteSigned```

Make sure pyinstaller and AlGrow requirements are installed
(in a regular command prompt)
```
pip install msvc-runtime 
pip install pyinstaller
pip install -r REQUIREMENTS.txt

```
Then run pyinstaller
```
pyinstaller --onefile --paths src/ --clean --noconfirm --log-level WARN 
--name algrow_0_5_0_win10 
--icon=bmp\icon.ico 
--add-data=bmp\logo.png:.\bmp\ 
--add-data=venv\lib\site-packages\open3d\resources:.\open3d\resources
algrow.py
```

## Install
### From wheel
Download 
  - get the latest [dist](https://github.com/marcusmchale/algrow/dist)
Install
```pip install dist/algrow-0.3-py3-none-any.whl```
Run
```./algrow.py```
### From source
Download
```git clone https://github.com/marcusmchale/algrow```
Install requirements
```pip install -r REQUIREMENTS.txt```
Run
```./algrow.py```

#### Icon
to prepare icon file in linux environment with imagemagick installed:
```convert -density 384 icon.svg -define icon:auto-resize icon.ico```
