# Python and most build deps
sudo apt-get install -y build-essential autoconf libtool pkg-config \
    python3-dev python3-pip python3-numpy git flex bison libbz2-dev

# recent cmake version
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update && apt-get --allow-unauthenticated install -y \
    cmake \
    kitware-archive-keyring

python3 -m venv venv
source venv/bin/activate

# install NLE
pip install nle

# install MiniHack
pip install minihack

# update `render` method to visualize pixels
pip install ./utils/

# other dependencies
pip install torchvision
pip install matplotlib
pip install PyQt5