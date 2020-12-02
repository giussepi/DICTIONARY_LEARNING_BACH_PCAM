#!/bin/bash
#
# Deletes all .pyc files from the project
#

find . -name "*.pyc" -exec rm -f {} \;
