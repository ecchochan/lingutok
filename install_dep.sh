#!/bin/bash

apt install autoconf libtool opencc cmake doxygen -y

git submodule update --init --recursive

cd marisa-trie
autoreconf -i
./configure
make
# make install

# cp $PWD/lib/marisa/.libs/libmarisa.so /usr/lib/libmarisa.so
# cp $PWD/lib/marisa/.libs/libmarisa.so.0 /usr/lib/libmarisa.so.0
# ln -s /usr/lib/libmarisa.so /usr/lib/liblibmarisa-trie.so
# ln -s /usr/lib/libmarisa.so.0 /usr/lib/liblibmarisa-trie.so.0
# ln -s /usr/lib/libmarisa.so /usr/lib/liblibmarisa.so
# ln -s /usr/lib/libmarisa.so.0 /usr/lib/liblibmarisa.so.0
# export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH




cd utf8proc
mkdir build
cd build
cmake ..
# make install
cp libutf8proc.a /usr/lib/liblibutf8proc.a

cd ..



cd OpenCC
make
# make install
# cp $PWD/build/rel/src/libopencc.so /usr/lib/libopencc.so
# ln -s /usr/lib/libopencc.so /usr/lib/liblibopencc.so

cd ..


cp -f OpenCC/build/rel/data/*.ocd2 lingutok/resources/opencc_config/
cp -f OpenCC/build/rel/data/*.txt lingutok/resources/opencc_config/
cp -f OpenCC/data/config/* lingutok/resources/opencc_config/ 

