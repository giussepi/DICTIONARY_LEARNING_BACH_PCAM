#!/bin/bash
#
# Installs requirements.txt
#
# Clones/updates and installs third party repositories and makes them available as modules
# of dl_algorithms folder
#

echo "Installing requirements.txt"

pip install -r requirements.txt

echo "requirements.txt successfully installed"


echo "Installing third-party repositories"

cd dl_algorithms

repository_url="https://github.com/giussepi/LC-KSVD.git"
repository_folder="LC-KSVD"

if [ ! -d $repository_folder ]; then
    git clone $repository_url
    pip install -r $repository_folder/requirements.txt
    touch $repository_folder/__init__.py
else
    cd $repository_folder
    git pull --rebase origin master
    pip install -r requirements.txt
    cd ..
fi

echo "Third-party repositories installed/updated successfully"
