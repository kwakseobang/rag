#!/bin/bash

# 시스템 패키지 설치 (distutils 등)
sudo apt-get update
sudo apt-get install -y python3-distutils
pip install --upgrade pip
# Python 패키지 설치
pip install -r requirements.txt
