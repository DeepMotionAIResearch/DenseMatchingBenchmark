#!/usr/bin/env bash

if [[ $1 == 'spn' || $1 == 'all' ]]
then

    cd dmb/ops/spn/

    python setup.py clean
    rm -rf build
    rm -r dist
    rm -r *.egg-info

    if [[ $2 == 'install' ]]
    then
        python setup.py install
    else
        python setup.py build develop
    fi

    cd ../../../

    echo "*********************************************************************"
    echo "                         SPN installed!"
    echo "*********************************************************************"

fi


if [[ $1 == 'dmb' || $1 == 'all' ]]
then

    python setup.py clean
    rm -r build
    rm -r dist
    rm -r *.egg-info

    if [[ $2 == 'install' ]]
    then
        python setup.py install
    else
        python setup.py build develop
    fi

    echo "*********************************************************************"
    echo "                         dmb installed!"
    echo "*********************************************************************"

fi


echo "*********************************************************************"
echo "                Dense Matching Benchmark Installed!"
echo "*********************************************************************"

