#!/usr/bin/env bash

if [[ $1 == 'GANet' || $1 == 'all' ]]
then
    echo "GANet hasn't integrated into dmb! Nothing installed."
#    cd dense_matching_benchmark/dmb/modeling/stereo/cost_aggregation/utils/libGANet/
#
#    python setup.py clean
#
#    if [[ $2 == 'install' ]]
#    then
#        python setup.py install
#    else
#        rm -rf build
#        python setup.py build develop
#        cp -r build/lib* build/lib
#    fi
#    echo "GANet lib installed!"
#
#    cd ../../../../../../../
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
    echo "dmb installed!"

fi

echo "Dense Matching Benchmark Installed!"