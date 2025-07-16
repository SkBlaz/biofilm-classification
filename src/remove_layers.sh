#!/bin/bash
tmp=${1##*_}
num=${tmp%.*}

target=${2:-21}

if [ $num = $2 ]; then
    remove=`seq -s, 1 2 $((num-1))`

    echo "Removing layers $remove from $1"

    name=${1%_*}

    convert $1 -delete $remove ${name}_$(((num + 1)/2)).tif >/dev/null 2>/dev/null
fi