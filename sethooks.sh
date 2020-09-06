#!/bin/sh
ls -1 hooks | xargs -I % sh -c 'ln -s $(pwd)/hooks/% $(pwd)/.git/hooks/'
