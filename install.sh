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
repository_folder_renamed="lc_ksvd"

if [ ! -d $repository_folder_renamed ]; then
    git clone $repository_url
    mv $repository_folder $repository_folder_renamed
    cd $repository_folder_renamed
    touch __init__.py
else
    cd $repository_folder_renamed
    git pull --rebase origin master
fi

pip install -r requirements.txt
cd ..


echo "Third-party repositories installed/updated successfully"
