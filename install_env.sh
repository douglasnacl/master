#!/bin/bash
echo """
    Instalação de ambiente de desenvolvimento
        > Autor: Douglas de Oliveira
        > Data: 2023-03-15
        > Versão: 0.0.1
        :param cuda: Instalação de CUDA e CUDNN
    """

echo 'Inicializando configuração' 

if [ $# -gt 0 ] && [ "$1" == "cuda" ]; then
    if [ -f /etc/arch-release ]; then
        echo "Arch Linux"
        echo "pacman -S cuda cudnn"
        pacman -S qt5-base qt5-x11extras libxcb xcb-util xcb-util-keysyms xcb-util-image xcb-util-wm xcb-util-cursor
    elif [ -f /etc/debian_version ]; then
        echo "Debian"
        echo "apt-get install cuda cudnn"
    elif [ -f /etc/lsb-release ] && grep -q 'Ubuntu' /etc/lsb-release; then
        echo "Ubuntu"
        echo "apt-get install cuda cudnn"
    else
        echo "Unknown distribution"
    fi
else
echo """
    Quer realizar a instalação de CUDA e CUDNN?
        > Se sim, para instalar CUDA e CUDNN, execute o script com o argumento 'cuda'
    """
fi

echo 'Configure conda environment libraries' 
# pip install tensorflow-gpu==2.10.0
# pip install bitfinex-tencars==0.0.3
# pip install tensorflow-gpu==2.10.0
# pip install opencv-python==4.7.0.72
# pip install tensorboardx==2.5.1
# pip install mplfinance==0.12.9b5
# pip install matplotlib==3.2.2
# pip install seaborn==0.11.2
# pip install pandas==1.3.5
# pip install numpy==1.24.2
# # !pip install numba==0.56.4
# pip install ta==0.10.2