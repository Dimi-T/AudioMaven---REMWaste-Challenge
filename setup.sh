#!/bin/bash

echo Fetching Updates...
exec > /dev/null
sudo apt --fix-broken install && sudo apt-get -y upgrade
exec > /dev/tty
echo Done!

echo Installing Dependencies...
exec > /dev/null
sudo apt-get -y install python3.9
sudo apt-get -y install python3-pip
exec > /dev/tty
echo Done!


echo Installing Cuda...
exec > /dev/null

# Taken from the official Nvidia CUDA installation guide
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda-repo-ubuntu2004-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-9-local_12.9.0-575.51.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-9-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9

sudo apt-get install -y cuda-drivers

exec > /dev/tty
echo Done!

echo Installing Python Packages...
exec > /dev/null
pip3 install -qq -r requirements.txt
exec > /dev/tty
echo Done!


echo "A system reboot is required for CUDA drivers to be properly loaded."
read -p "Would you like to reboot now? (y/n): " reboot_choice

if [[ "$reboot_choice" == "y" || "$reboot_choice" == "Y" ]]; then
    echo "Rebooting the system..."
    sudo reboot
else
    echo "Please reboot the system manually later to apply changes."
fi