# Install OpenAI and Dependencies


pip install numpy

pip install --upgrade tensorflow

#GYM
git clone https://github.com/openai/gym.git
cd gym
pip install -e .

#For Full Version
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig

#install atari etc
pip install -e '.[atari]'

# to install pygame for openai gym
pip install pygame

git clone https://github.com/ntasfi/PyGame-Learning-Environment.git
cd PyGame-Learning-Environment/
pip install -e .

git clone https://github.com/lusob/gym-ple.git
cd gym-ple/
pip install -e .


#Install Universe
pip install numpy
sudo apt-get install golang libjpeg-turbo8-dev make

git clone https://github.com/openai/universe.git
cd universe
pip install -e .

#Install docker from their website


# OpenGL rendering support 
sudo apt-get install libx11-dev libxcursor-dev libxrandr-dev libxinerama-dev libxi-dev libxxf86vm-dev libgl1-mesa-dev mesa-common-dev
git clone https://github.com/openai/go-vncdriver.git
cd go-vncdriver
python build.py
pip install -e .
