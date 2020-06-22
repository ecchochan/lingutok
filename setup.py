#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from setuptools import setup, find_packages, Extension
import glob
import itertools
import os


MARISA_ROOT_DIR = "marisa-trie"
MARISA_SOURCE_DIR = os.path.join(MARISA_ROOT_DIR, "lib")
MARISA_INCLUDE_DIR = os.path.join(MARISA_ROOT_DIR, "include")
MARISA_FILES = [
    "marisa/*.cc",
    "marisa/grimoire.cc",
    "marisa/grimoire/io/*.cc",
    "marisa/grimoire/trie/*.cc",
    "marisa/grimoire/vector/*.cc",
]

MARISA_FILES[:] = itertools.chain(
    *(glob.glob(os.path.join(MARISA_SOURCE_DIR, path))
    for path in MARISA_FILES))


ext_modules = [
    Extension("LinguisticTokenizer", [
        "src/agent.cpp",
        "src/base.cpp",
        "src/iostream.cpp",
        "src/key.cpp",
        "src/keyset.cpp",
        "src/LinguisticTokenizer.cpp",
        "src/query.cpp",
        "src/std_iostream.cpp",
        "src/trie.cpp"
    ], 
    include_dirs=[MARISA_INCLUDE_DIR, "utf8proc"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],),
]

setup(
    name='LinguisticTokenizer',
    version='0.0.1',
    description='Linguistic Tokenizer',
    libraries=[("libmarisa-trie", {
        "sources": MARISA_FILES,
        "include_dirs": [MARISA_SOURCE_DIR, MARISA_INCLUDE_DIR]
    }), ("libutf8proc", {
        "sources": ["utf8proc/utf8proc.c"],
        "include_dirs": ["utf8proc"]
    })],
    ext_modules=ext_modules,
    package_data={'LinguisticTokenizer': ['resources/*']}
)




