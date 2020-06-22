**An LinguisticTokenizer for English**



```bash

sudo apt install autoconf libtool -y


cd LinguisticTokenizer
git submodule update --init --recursive

cd marisa-trie
autoreconf -i
./configure
make
sudo make install

sudo cp $PWD/lib/marisa/.libs/libmarisa.so /usr/lib/liblibmarisa-trie.so
sudo cp $PWD/lib/marisa/.libs/libmarisa.so.0 /usr/lib/liblibmarisa-trie.so.0
sudo ln -s /usr/lib/liblibmarisa-trie.so /usr/lib/liblibmarisa.so
sudo ln -s /usr/lib/liblibmarisa-trie.so.0 /usr/lib/liblibmarisa.so.0
sudo ln -s /usr/lib/liblibmarisa-trie.so /usr/lib/libmarisa.so
sudo ln -s /usr/lib/liblibmarisa-trie.so.0 /usr/lib/libmarisa.so.0
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH



cd utf8proc
mkdir build
cd build
cmake ..
sudo make install
sudo cp libutf8proc.a /usr/lib/liblibutf8proc.a
export LD_LIBRARY_PATH=/usr/lib:$LD_LIBRARY_PATH
cd ..




cd ..

./update_cpp.sh

./update_cpp.sh && sudo python3 setup.py develop

```