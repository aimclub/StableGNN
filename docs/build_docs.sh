#!/bin/bash
set -x
cd /docs
###################
# INSTALL DEPENDS #
###################

apt-get update
apt-get -y install git rsync python3-sphinx python3-sphinx-rtd-theme python3-pip
sphinx-apidoc -o ./docs ./stable_ggn
make -C docs clean
make -C docs html
chmod -R 7777 docs
exit 0
