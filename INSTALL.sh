#!/usr/bin/env bash


if [[ $1 == 'GANet' || $1 == 'all' ]]
then
    cd dmb/ops/libGANet/

    python setup.py clean

    if [[ $2 == 'install' ]]
    then
        python setup.py install
    else
        rm -rf build
        python setup.py build develop
        cp -r build/lib* build/lib
    fi

    cd ../../../

    echo "*********************************************************************"
    echo "                         GANet installed!"
    echo "*********************************************************************"

fi


if [[ $1 == 'spn' || $1 == 'all' ]]
then

    cd dmb/ops/spn/

    python setup.py clean

    if [[ $2 == 'install' ]]
    then
        python setup.py install
    else
        rm -rf build
        python setup.py build develop
        cp -r build/lib* build/lib
    fi

    cd ../../../

    echo "*********************************************************************"
    echo "                         SPN installed!"
    echo "*********************************************************************"

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
        cp -r build/lib* build/lib
    fi

    echo "*********************************************************************"
    echo "                         dmb installed!"
    echo "*********************************************************************"

fi


echo "*********************************************************************"
echo "                Dense Matching Benchmark Installed!"
echo "*********************************************************************"

