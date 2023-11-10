#!/bin/bash

# Downloading train dataset
if [ -f "/VTON/ACGPN_traindata.zip" ]; then
  echo "ACGPN_traindata existed. Skip downloading"
else
  echo "Downloading ACGPN_traindata..."
  gdown "https://drive.google.com/uc?id=1lHNujZIq6KVeGOOdwnOXVCSR5E7Kv6xv"  # train dataset
fi

# Unzipping train dataset
if [ -d "/VTON/ACGPN_traindata" ]; then
  echo "ACGPN_traindata existed. Skip unzipping"
else
  echo "Unzipping ACGPN_traindata..."
  unzip ACGPN_traindata.zip -d /VTON/ACGPN_traindata
fi

# Downloading test dataset
if [ -f "/VTON/Data_preprocessing.zip" ]; then
  echo "ACGPN_testdata existed. Skip downloading"
else
  echo "Downloading ACGPN_testdata..."
  gdown "https://drive.google.com/uc?id=1tE7hcVFm8Td8kRh5iYRBSDFdvZIkbUIR"  # test dataset
fi

# Unzipping test dataset
if [ -d "/VTON/ACGPN_testdata" ]; then
  echo "ACGPN_testdata existed. Skip unzipping"
else
  echo "Unzipping ACGPN_traindata..."
  unzip Data_preprocessing.zip -d /VTON/ACGPN_testdata
fi

# Download & unzip VGG pre-trained model
if [ -f "/VTON/ACGPN_TrainData/models/vgg19-dcbb9e9d.pth" ]; then
  echo "vgg19-dcbb9e9d.pth existed. Skip downloading"
else
  echo "Downloading vgg19-dcbb9e9d.pth to /VTON/ACGPN_TrainData/models/..."
  wget -O /VTON/ACGPN_TrainData/models/vgg19-dcbb9e9d.pth https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
fi

# Download ACGPN checkpoint for transfer-learning
if [ -f "/VTON/ACGPN_TrainData/models/ACGPN_checkpoints.zip" ]; then
  echo "ACGPN_checkpoints existed. Skip downloading"
else
  echo "Downloading ACGPN_checkpoints..."
  gdown "https://drive.google.com/uc?id=1UWT6esQIU_d4tUm8cjxDKMhB8joQbrFx"
fi

# Unzipping ACGPN checkpoint
if [ -d "/VTON/ACGPN_checkpoints" ]; then
  echo "ACGPN_checkpoints existed. Skip unzipping"
else
  echo "Unzipping ACGPN_checkpoints..."
  unzip ACGPN_checkpoints.zip -d /VTON/ACGPN_checkpoints
fi
