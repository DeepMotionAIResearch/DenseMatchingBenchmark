#!/usr/bin/env bash

if [[ $1 == 'GANet' || $1 == 'all' ]]
then
    echo "GANet hasn't integrated into dmb! Nothing installed."

fi

if [[ $1 == 'dmb' || $1 == 'all' ]]
then

    python setup.py clean

    if [[ $2 == 'install' ]]
    then
        python setup.py install
    else
        rm -r build
        python setup.py build develop
    fi
    echo "dmb installed!"

fi

echo "Dense Matching Benchmark Installed!"