#!/bin/bash

apt install autoconf libtool -y

git submodule update --init --recursive

cd marisa-trie
autoreconf -i
./configure
make
make install

ln -s $PWD/lib/marisa/.libs/libmarisa.so /usr/lib/liblibmarisa-trie.so
ln -s $PWD/lib/marisa/.libs/libmarisa.so.0 /usr/lib/liblibmarisa-trie.so.0
ln -s $PWD/lib/marisa/.libs/libmarisa.so /usr/lib/liblibmarisa.so
ln -s $PWD/lib/marisa/.libs/libmarisa.so.0 /usr/lib/liblibmarisa.so.0
ln -s $PWD/lib/marisa/.libs/libmarisa.so /usr/lib/libmarisa.so
ln -s $PWD/lib/marisa/.libs/libmarisa.so.0 /usr/lib/libmarisa.so.0
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH

cd ..

./update_cpp.sh

