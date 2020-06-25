#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# cython: profile=False, embedsignature=True, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=2, language=c++
#distutils: language = c++
# distutils: extra_compile_args = -openmp
# distutils: extra_link_args = -openmp

from __future__ import unicode_literals

from std_iostream cimport stringstream, istream, ostream
from libc.string cimport strncmp    
cimport keyset
cimport key
cimport agent
cimport trie
cimport iostream
cimport base
from utf8proc cimport utf8proc_NFKD_strip

from lrucache cimport lru_cache


import itertools
import struct
import warnings
from libcpp.utility cimport pair

import numpy as pynp
cimport numpy as np

from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy 
from libc.stdio cimport printf




cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T,ALLOCATOR=*] nogil:
        ctypedef T value_type
        ctypedef ALLOCATOR allocator_type

        # these should really be allocator_type.size_type and
        # allocator_type.difference_type to be true to the C++ definition
        # but cython doesn't support deferred access on template arguments
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        cppclass iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            iterator operator+(size_type) nogil
            iterator operator-(size_type) nogil
            difference_type operator-(iterator) nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
            bint operator<(iterator) nogil
            bint operator>(iterator) nogil
            bint operator<=(iterator) nogil
            bint operator>=(iterator) nogil
        cppclass reverse_iterator:
            T& operator*() nogil
            reverse_iterator operator++() nogil
            reverse_iterator operator--() nogil
            reverse_iterator operator+(size_type) nogil
            reverse_iterator operator-(size_type) nogil
            difference_type operator-(reverse_iterator) nogil
            bint operator==(reverse_iterator) nogil
            bint operator!=(reverse_iterator) nogil
            bint operator<(reverse_iterator) nogil
            bint operator>(reverse_iterator) nogil
            bint operator<=(reverse_iterator) nogil
            bint operator>=(reverse_iterator) nogil
        cppclass const_iterator(iterator):
            pass
        cppclass const_reverse_iterator(reverse_iterator):
            pass
        vector() nogil
        vector(vector&) nogil
        vector(size_type) nogil
        vector(size_type, T&) nogil
        #vector[input_iterator](input_iterator, input_iterator)
        T& operator[](size_type) nogil
        #vector& operator=(vector&)
        bint operator==(vector&, vector&) nogil
        bint operator!=(vector&, vector&) nogil
        bint operator<(vector&, vector&) nogil
        bint operator>(vector&, vector&) nogil
        bint operator<=(vector&, vector&) nogil
        bint operator>=(vector&, vector&) nogil
        void assign(size_type, const T&) nogil
        void assign[input_iterator](input_iterator, input_iterator) nogil
        T& at(size_type) nogil
        T& back() nogil
        iterator begin() nogil
        const_iterator const_begin "begin"() nogil
        size_type capacity() nogil
        void clear() nogil
        bint empty() nogil
        iterator end() nogil
        const_iterator const_end "end"() nogil
        iterator erase(iterator) nogil
        iterator erase(iterator, iterator) nogil
        T& front() nogil
        iterator insert(iterator, const T&) nogil
        iterator insert(iterator, size_type, const T&) nogil
        iterator insert[Iter](iterator, Iter, Iter) nogil
        size_type max_size() nogil
        void pop_back() nogil
        void push_back(T&)  nogil
        reverse_iterator rbegin() nogil
        const_reverse_iterator const_rbegin "crbegin"() nogil
        reverse_iterator rend() nogil
        const_reverse_iterator const_rend "crend"() nogil
        void reserve(size_type) nogil
        void resize(size_type) nogil
        void resize(size_type, T&) nogil
        size_type size() nogil
        void swap(vector&) nogil

        # C++11 methods
        T* data() nogil
        const T* const_data "data"() nogil
        void shrink_to_fit() nogil
        
        


cdef extern from "<unordered_map>" namespace "std":
    cdef cppclass unordered_map[T, U] nogil:
        cppclass iterator:
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
            pair[T, U]& operator*() nogil
        unordered_map() nogil
        unordered_map(unordered_map&) nogil
        U& operator[](T&) nogil
        U& at(T&) nogil
        iterator find(T&) nogil 
        iterator begin() nogil
        iterator end() nogil
        bint empty() nogil
        void clear() nogil
        size_t count(T&) nogil
        size_t size() nogil

cdef extern from "<unordered_set>" namespace "std" nogil:
    cdef cppclass unordered_set[T,HASH=*,PRED=*,ALLOCATOR=*] nogil:
        cppclass iterator:
            T& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        unordered_set() except +
        unordered_set(unordered_set&) except +
        iterator find(T&) nogil
        iterator begin() nogil
        iterator end() nogil
        bint empty() nogil
        void clear() nogil
        size_t count(T&) nogil
        size_t max_size() nogil
        size_t size() nogil
        pair[iterator, bint] insert(T&)



try:
    from itertools import izip
except ImportError:
    izip = zip


DEFAULT_CACHE = base.MARISA_DEFAULT_CACHE
HUGE_CACHE = base.MARISA_HUGE_CACHE
LARGE_CACHE = base.MARISA_LARGE_CACHE
NORMAL_CACHE = base.MARISA_NORMAL_CACHE
SMALL_CACHE = base.MARISA_SMALL_CACHE
TINY_CACHE = base.MARISA_TINY_CACHE

MIN_NUM_TRIES = base.MARISA_MIN_NUM_TRIES
MAX_NUM_TRIES = base.MARISA_MAX_NUM_TRIES
DEFAULT_NUM_TRIES = base.MARISA_DEFAULT_NUM_TRIES

# MARISA_TEXT_TAIL merges last labels as zero-terminated strings. So, it is
# available if and only if the last labels do not contain a NULL character.
# If MARISA_TEXT_TAIL is specified and a NULL character exists in the last
# labels, the setting is automatically switched to MARISA_BINARY_TAIL.
TEXT_TAIL = base.MARISA_TEXT_TAIL

# MARISA_BINARY_TAIL also merges last labels but as byte sequences. It uses
# a bit vector to detect the end of a sequence, instead of NULL characters.
# So, MARISA_BINARY_TAIL requires a larger space if the average length of
# labels is greater than 8.
BINARY_TAIL = base.MARISA_BINARY_TAIL
DEFAULT_TAIL = base.MARISA_DEFAULT_TAIL


# MARISA_LABEL_ORDER arranges nodes in ascending label order.
# MARISA_LABEL_ORDER is useful if an application needs to predict keys in
# label order.
LABEL_ORDER = base.MARISA_LABEL_ORDER

# MARISA_WEIGHT_ORDER arranges nodes in descending weight order.
# MARISA_WEIGHT_ORDER is generally a better choice because it enables faster
# matching.
WEIGHT_ORDER = base.MARISA_WEIGHT_ORDER
DEFAULT_ORDER = base.MARISA_DEFAULT_ORDER


cdef class _Trie:
    cdef trie.Trie* _trie

    cdef bytes _encode_key(self, key):
        return key

    cdef _get_key(self, agent.Agent& ag):
        return ag.key().ptr()[:ag.key().length()]

    cdef int _get_id(self, agent.Agent& ag) nogil:
        return ag.key().id()

    def __init__(self, arg=None, num_tries=DEFAULT_NUM_TRIES, binary=False,
                 cache_size=DEFAULT_CACHE, order=DEFAULT_ORDER, weights=None):
        """
        ``arg`` can be one of the following:

        * an iterable with bytes keys;
        * None (if you're going to load a trie later).

        Pass a ``weights`` iterable with expected lookup frequencies
        to optimize lookup and prefix search speed.
        """

        if self._trie:
            return
        self._trie = new trie.Trie()

        byte_keys = (self._encode_key(key) for key in (arg or []))

        self._build(
            byte_keys,
            weights,
            num_tries=num_tries,
            binary=binary,
            cache_size=cache_size,
            order=order
        )

    def __dealloc__(self):
        if self._trie:
            del self._trie

    def _config_flags(self, num_tries=DEFAULT_NUM_TRIES, binary=False,
                      cache_size=DEFAULT_CACHE, order=DEFAULT_ORDER):
        if not MIN_NUM_TRIES <= num_tries <= MAX_NUM_TRIES:
            raise ValueError(
                "num_tries (which is %d) must be between between %d and %d" %
                (num_tries, MIN_NUM_TRIES, MAX_NUM_TRIES))

        binary_flag = BINARY_TAIL if binary else TEXT_TAIL
        return num_tries | binary_flag | cache_size | order

    def _build(self, byte_keys, weights=None, **options):
        if weights is None:
            weights = itertools.repeat(1.0)

        cdef char* data
        cdef float weight
        cdef keyset.Keyset *ks = new keyset.Keyset()

        try:
            for key, weight in izip(byte_keys, weights):
                ks.push_back(<char *>key, len(key), weight)
            self._trie.build(ks[0], self._config_flags(**options))
        finally:
            del ks

    def __richcmp__(self, other, int op):
        if op == 2:    # ==
            if other is self:
                return True
            elif not isinstance(other, _Trie):
                return False

            return (<_Trie>self)._equals(other)
        elif op == 3:  # !=
            return not (self == other)

        raise TypeError("unorderable types: {0} and {1}".format(
            self.__class__, other.__class__))

    cdef bint _equals(self, _Trie other) nogil:
        cdef int num_keys = self._trie.num_keys()
        cdef base.NodeOrder node_order = self._trie.node_order()
        if (other._trie.num_keys() != num_keys or
            other._trie.node_order() != node_order):
            return False

        cdef agent.Agent ag1, ag2
        ag1.set_query(b"")
        ag2.set_query(b"")
        cdef int i
        cdef key.Key key1, key2
        for i in range(num_keys):
            self._trie.predictive_search(ag1)
            other._trie.predictive_search(ag2)
            key1 = ag1.key()
            key2 = ag2.key()
            if (key1.length() != key2.length() or
                strncmp(key1.ptr(), key2.ptr(), key1.length()) != 0):
                return False
        return True

    def __iter__(self):
        return self.iterkeys()

    def __len__(self):
        return self._trie.num_keys()

    def __contains__(self, key):
        cdef bytes _key = self._encode_key(key)
        return self._contains(_key)

    cdef bint _contains(self, bytes key):
        cdef agent.Agent ag
        ag.set_query(key, len(key))
        return self._trie.lookup(ag)

    def read(self, f):
        """Read a trie from an open file.

        :param file f: a "real" on-disk file object. Passing a *file-like*
                       object would result in an error.

        .. deprecated:: 0.7.3

           The method will be removed in version 0.8.0. Please use
           :meth:`load` instead.
        """
        warnings.warn("Trie.save is deprecated and will "
                      "be removed in marisa_trie 0.8.0. Please use "
                      "Trie.load instead.", DeprecationWarning)
        self._trie.read(f.fileno())
        return self

    def write(self, f):
        """Write a trie to an open file.

        :param file f: a "real" on-disk file object. Passing a *file-like*
                       object would result in an error.

        .. deprecated:: 0.7.3

           The method will be removed in version 0.8.0. Please use
           :meth:`save` instead.
        """
        warnings.warn("Trie.write is deprecated and will "
                      "be removed in marisa_trie 0.8.0. Please use "
                      "Trie.save instead.", DeprecationWarning)
        self._trie.write(f.fileno())

    def save(self, path):
        """Save a trie to a specified path."""
        with open(path, 'w') as f:
            self._trie.write(f.fileno())

    def load(self, path):
        """Load a trie from a specified path."""
        with open(path, 'r') as f:
            self._trie.read(f.fileno())
        return self

    cpdef bytes tobytes(self) except +:
        """Return raw trie content as bytes."""
        cdef stringstream stream
        iostream.write((<ostream *> &stream)[0], self._trie[0])
        cdef bytes res = stream.str()
        return res

    cpdef frombytes(self, bytes data) except +:
        """Load a trie from raw bytes generated by :meth:`tobytes`."""
        cdef stringstream* stream = new stringstream(data)
        try:
            iostream.read((<istream *> stream)[0], self._trie)
        finally:
            del stream
        return self

    def __reduce__(self):
        return self.__class__, (), self.tobytes()

    __setstate__ = frombytes

    def mmap(self, path):
        """Memory map the content of a trie stored in a file.

        This allows to query trie without loading it fully in memory.
        """
        import sys
        str_path = path.encode(sys.getfilesystemencoding())
        cdef char* c_path = str_path
        self._trie.mmap(c_path)
        return self

    def iterkeys(self, prefix=None):
        """
        Return an iterator over trie keys starting with a given ``prefix``.
        """
        cdef agent.Agent ag
        cdef bytes b_prefix = b''
        if prefix is not None:
            b_prefix = self._encode_key(prefix)
        ag.set_query(b_prefix, len(b_prefix))

        while self._trie.predictive_search(ag):
            yield self._get_key(ag)

    cpdef list keys(self, prefix=None):
        """Return a list of trie keys starting with a given ``prefix``."""
        # non-generator inlined version of iterkeys()
        cdef list res = []
        cdef bytes b_prefix = b''
        if prefix is not None:
            b_prefix = self._encode_key(prefix)
        cdef agent.Agent ag
        ag.set_query(b_prefix, len(b_prefix))

        while self._trie.predictive_search(ag):
            res.append(self._get_key(ag))

        return res

    def has_keys_with_prefix(self, prefix=""):
        """
        Return ``True`` if any key in the trie begins with ``prefix``.

        .. deprecated:: 0.7.3

           The method will be removed in version 0.8.0. Please use
           :meth:`iterkeys` instead.
        """
        warnings.warn("Trie.has_keys_with_prefix is deprecated and will "
                      "be removed in marisa_trie 0.8.0. Please use "
                      "Trie.iterkeys instead.", DeprecationWarning)

        cdef agent.Agent ag
        cdef bytes b_prefix = self._encode_key(prefix)
        ag.set_query(b_prefix, len(b_prefix))
        return self._trie.predictive_search(ag)


cdef class _UnicodeKeyedTrie(_Trie):
    """
    MARISA-trie wrapper for unicode keys.
    """
    cdef bytes _encode_key(self, key):
        return key.encode('utf8')

    cdef _get_key(self, agent.Agent& ag):
        return <unicode>_Trie._get_key(self, ag).decode('utf8')


cdef class Trie(_Trie):
    """A trie mapping unicode keys to auto-generated unique IDs."""

    # key_id method is not in _Trie because it won't work for BytesTrie
    cpdef int key_id(self, unicode key) except -1:
        """Return an ID generated for a given ``key``.

        :raises KeyError: if key is not present in this trie.
        """
        cdef bytes _key = <bytes>key.encode('utf8')
        cdef int res = self._key_id(_key)
        if res == -1:
            raise KeyError(key)
        return res

    def __getitem__(self, unicode key):
        return self.key_id(key)

    def get(self, key, default=None):
        """
        Return an ID for a given ``key`` or ``default`` if ``key`` is
        not present in this trie.
        """
        cdef bytes b_key
        cdef int res

        if isinstance(key, unicode):
            b_key = <bytes>(<unicode>key).encode('utf8')
        else:
            b_key = key

        res = self._key_id(b_key)
        if res == -1:
            return default
        return res

    cpdef restore_key(self, int index):
        """Return a key corresponding to a given ID."""
        cdef agent.Agent ag
        ag.set_query(index)
        try:
            self._trie.reverse_lookup(ag)
        except KeyError:
            raise KeyError(index)
        return self._get_key(ag)

    cdef int _key_id(self, char* key):
        cdef bint res
        cdef agent.Agent ag
        ag.set_query(key)
        res = self._trie.lookup(ag)
        if not res:
            return -1
        return ag.key().id()

    def iter_prefixes(self, unicode key):
        """
        Return an iterator of all prefixes of a given key.
        """
        cdef bytes b_key = <bytes>key.encode('utf8')
        cdef agent.Agent ag
        ag.set_query(b_key)

        while self._trie.common_prefix_search(ag):
            yield self._get_key(ag)

    def prefixes(self, unicode key):
        """
        Return a list with all prefixes of a given key.
        """
        # this an inlined version of ``list(self.iter_prefixes(key))``

        cdef list res = []
        cdef bytes b_key = <bytes>key.encode('utf8')
        cdef agent.Agent ag
        ag.set_query(b_key)

        while self._trie.common_prefix_search(ag):
            res.append(self._get_key(ag))
        return res

    cdef vector[int] _prefixes_id(self, char* b_key, int length) nogil:
        """
        Return a list with all prefixes of a given key.
        """
        # this an inlined version of ``list(self.iter_prefixes(key))``

        cdef vector[int] res = vector[int]()
        cdef agent.Agent ag
        ag.set_query(b_key, length)
        res.reserve(8)

        while self._trie.common_prefix_search(ag):
            res.push_back(self._get_id(ag))
        return res

    def iteritems(self, unicode prefix=""):
        """
        Return an iterator over items that have a prefix ``prefix``.
        """
        cdef bytes b_prefix = <bytes>prefix.encode('utf8')
        cdef agent.Agent ag
        ag.set_query(b_prefix)

        while self._trie.predictive_search(ag):
            yield self._get_key(ag), ag.key().id()

    def items(self, unicode prefix=""):
        # inlined for speed
        cdef list res = []
        cdef bytes b_prefix = <bytes>prefix.encode('utf8')
        cdef agent.Agent ag
        ag.set_query(b_prefix)

        while self._trie.predictive_search(ag):
            res.append((self._get_key(ag), ag.key().id()))

        return res









cdef int MERGE_MODE_NORMAL = 0
cdef int MERGE_MODE_BOTH   = 1
cdef int MERGE_MODE_SINGLE = 2
    
cdef int PREFIX = 0
cdef int ROOT = 1
cdef int SUFFIX = 2
    
'''
cdef class Part():
    cdef int contents
    cdef int form
    cdef bint merge_mode

    def __init__(self, contents, form, merge_mode):
        self.contents = contents
        self.form = form
        self.merge_mode = merge_mode
    
cdef class ():
    cdef list contents
    cdef float score

    def __init__(self, contents, score):
        self.contents = contents
        self.score = score
'''

    
    

cdef float ROOT_PUNISHMENT = 0.5

'''
    e = 'abs'
    key = hash_string(e)                      # The key from the prefix
    bucket = get_bucket(key)
    bucket.push_back((vec_part, 0))
    trie_values[key] = bucket                 # what we want from prefixes

    ...

    for e in trie.prefixes('absolutely'):
        key = hash_string(e.decode())
        tire_key = tire._key_id(e.decode())   


    ...

    #  cdef vector[int] _prefixes_id(self, bytes key):

    cdef vector[int] prefixes_in_trie_ids = trie._prefixes_id('absolutely')

    cdef int trie_id
    for i in range(<int>prefixes_in_trie_ids.size()):
        trie_id = prefixes_in_trie_ids[i]     # Trie internal id

    ...

    
    cdef unordered_map[int, vector[Parts]] true_trie_values


    for e in trie:
        key = hash_string(e.decode())
        trie_id = trie[e]
        true_trie_values[trie_id] = trie_values[key]



'''


ctypedef (vector[int], float, int,  int          ) Parts
#           part_ids    score  last  len_part_max









from pkg_resources import resource_filename


package_name = 'LinguisticTokenizer'
import os.path
import json


def get_file(x):
    return resource_filename(package_name, 'resources/' + x)


cdef int SPECIAL_TOKEN_CLS  = 2
cdef int SPECIAL_TOKEN_SEP  = 3
cdef int SPECIAL_TOKEN_MASK = 5
cdef int SPECIAL_TOKEN_PAD  = 0
cdef int SPECIAL_TOKEN_UNK  = 1
cdef int SPECIAL_TOKEN_NL   = 10

cdef unordered_map[int, int*] replacements
cdef int* replacement_temp
cdef int L, i
fn = get_file('zh_char2str_mapping.txt')
with open(fn) as f:
    for line in f:
        line = line[:len(line)-1]
        if not line:
            continue
        splitted = line.split('\t')
        if len(splitted) != 2:
            continue
        a, b = splitted
        assert len(a) == 1
        L = len(b)
        replacement_temp = <int*> malloc(sizeof(int) * (L+1))
        replacement_temp[0] = L
        for i in range(L):
            replacement_temp[i+1] = ord(b[i])
            
        replacements[ord(a)] = replacement_temp


custom_mapping = {
    '\r': (ord(' '),),
    '\t': (ord(' '),),
}
for k, v in custom_mapping.items():
    L = len(v)
    replacement_temp = <int*> malloc(sizeof(int) * (L+1))
    replacement_temp[0] = L
    for i in range(L):
        replacement_temp[i+1] = v[i]
    replacements[ord(k)] = replacement_temp







'''
# Embedded in iter_unicode

SPECIAL_TOKENS = {
    '<s>',
    '</s>',
    '<mask>',
    '<pad>',
    '<unk>',
    '<nl>',
}
'''
cdef int _2_6 = 2**6
cdef int _2_12 = 2**12
cdef int _2_18 = 2**18
cdef int iter_unicode(const unsigned char[:] chars, vector[int]* bucket, bint keep_nl) nogil:
    cdef:
        size_t length = chars.shape[0]
        unsigned char lb, last_lb = 0
        size_t cursor = 0, repl_size, i,  en_char_length
        char size = 0
        int code
        int* repl
        
        size_t cased_count = 0
        
    bucket.reserve(length)
        
    while cursor < length:
        lb = chars[cursor]
        if (lb - 0xc2) > (0xf4-0xc2):
            return -1
        if lb < 0x80:
            size = 1
            code = lb
            
            if last_lb == 60: # '<' 
                # check <s>
                if lb == 115 and length - cursor >= 2:  # 3 - 1
                    if (
                        chars[cursor+1] == 62   # >
                    ): 
                        bucket.pop_back() # remove '<'
                        bucket.push_back(SPECIAL_TOKEN_CLS)
                        size = 2  # 3 - 1
                        cursor += size
                        last_lb = 0
                        continue
                     
                # check </s>
                elif lb == 47 and length - cursor >= 3:  # 4 - 1
                    if (
                        chars[cursor+1] == 115 and  # s
                        chars[cursor+2] == 62       # >
                    ): 
                        bucket.pop_back() # remove '<'
                        bucket.push_back(SPECIAL_TOKEN_SEP)
                        size = 3  # 4 - 1
                        cursor += size
                        last_lb = 0
                        continue
                        
                # check <mask>  (109, 97, 115, 107)
                elif lb == 109 and length - cursor >= 5:  # 6 - 1
                    if (
                        chars[cursor+1] == 97 and   # a
                        chars[cursor+2] == 115 and  # s
                        chars[cursor+3] == 107 and  # k
                        chars[cursor+4] == 62       # >
                    ): 
                        bucket.pop_back() # remove '<'
                        bucket.push_back(SPECIAL_TOKEN_MASK)
                        size = 5  # 6 - 1
                        cursor += size
                        last_lb = 0
                        continue
                    
                # check <pad>  (112, 97, 100)
                elif lb == 112 and length - cursor >= 4:  # 5 - 1
                    if (
                        chars[cursor+1] == 97 and   # a
                        chars[cursor+2] == 100 and  # d
                        chars[cursor+3] == 62       # >
                    ): 
                        bucket.pop_back() # remove '<'
                        bucket.push_back(SPECIAL_TOKEN_PAD)
                        size = 4  # 5 - 1
                        cursor += size
                        last_lb = 0
                        continue
                    
                # check <unk>  (117, 110, 107)
                elif lb == 117 and length - cursor >= 4:  # 5 - 1
                    if (
                        chars[cursor+1] == 110 and   # a
                        chars[cursor+2] == 107 and  # d
                        chars[cursor+3] == 62       # >
                    ): 
                        bucket.pop_back() # remove '<'
                        bucket.push_back(SPECIAL_TOKEN_UNK)
                        size = 4  # 5 - 1
                        cursor += size
                        last_lb = 0
                        continue
                    
                # check <nl>  (110, 108)
                elif lb == 110 and length - cursor >= 3:  # 4 - 1
                    if (
                        chars[cursor+1] == 108 and   # l
                        chars[cursor+2] == 62        # >
                    ): 
                        bucket.pop_back() # remove '<'
                        bucket.push_back(SPECIAL_TOKEN_NL)
                        size = 3  # 4 - 1
                        cursor += size
                        last_lb = 0
                        continue
                    
            
        elif lb < 0xE0:
            size = 2
            if cursor + size > length:
                return -1
            
            #code = (a & 0b00000111)*2**6  + (b & 0b00111111)
            #code = (lb & 0b00000111)*_2_6 + (chars[cursor+1] & 0b00111111)
            code = ((lb & 0x1f)<<6) | (chars[cursor+1] & 0x3f)
            
            
        elif lb < 0xF0:
            size = 3
            if cursor + size > length:
                return -1
            
            #code = (a & 0b00000111)*2**12  + (b & 0b00111111) * 2**6 + (c & 0b00111111 )
            #code = (lb & 0b00000111)*_2_12 + (chars[cursor+1] & 0b00111111)*_2_6 + (chars[cursor+2] & 0b00111111)
            code = ((lb & 0xf)<<12) | ((chars[cursor+1] & 0x3f)<<6) | (chars[cursor+2] & 0x3f);
            
        elif ( lb & 0xF8 ) == 0xF0:
            size = 4
            if cursor + size > length:
                return -1
            
            #code = (a & 0b00000111)*2**18  + (b & 0b00111111) * 2**12 + (c & 0b00111111 ) * 2**6 + (d & 0b00111111 )
            #code = (lb & 0b00000111)*_2_18 + (chars[cursor+1] & 0b00111111)*_2_12 + (chars[cursor+2] & 0b00111111)*_2_6 + (chars[cursor+3] & 0b00111111)
            code = ((lb & 7)<<18) | ((chars[cursor+1] & 0x3f)<<12) | ((chars[cursor+2] & 0x3f)<<6) | (chars[cursor+3] & 0x3f)
            
        else:
            return -2
            
            
        # Normalize some unicode
        if replacements.find(code) != replacements.end():
            repl = replacements[code]+1
            repl_size = repl[-1]
            
        else:
            repl = &code
            repl_size = 1
            
        for i in range(repl_size):
            code = repl[i]
            # check a-z (97-122) or A-Z (65-90) or ' (39)
            if (code >= 97 and code <= 122) or code == 39:
                en_char_length += 1
            elif (code >= 65 and code <= 90):
                cased_count += 1
                en_char_length += 1
                
            elif code == 10:
                cased_count = 0
                en_char_length = 0
                if keep_nl:
                    bucket.push_back(SPECIAL_TOKEN_NL)
                    
            elif code == 32:
                cased_count = 0
                en_char_length = 0
                
            else:
                if en_char_length > 0:
                    # Situation:   `::!! english `
                    #                           ^ cursor
                    # en_char_length: 7
                    #
                    # char span:      [cusor - 7, cursor-1]  (i.e. inclusive)
                    #
                    
                        
                    pass
                        
                    
                cased_count = 0
                en_char_length = 0
                bucket.push_back(code)
        cursor += size
        last_lb = lb
        

    return 1

def py_iter_unicode(str text, bint keep_nl=False):
    cdef:
        vector[int] bucket
        int code
    code = iter_unicode(text.encode('utf-8'), &bucket, keep_nl)
    if code == -1:
        raise Exception ("unexpected end of data")
    elif code == -2:
        raise Exception ('can\'t decode bytes')
    return bucket

def test():
    text = "[]"
    assert py_iter_unicode(text) == [91, 93], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "我有  嗎"
    assert py_iter_unicode(text) == [25105, 26377, 21966], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "我有  嗎 ⒉"
    assert py_iter_unicode(text) == [25105, 26377, 21966, 50, 46], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "我有  嗎 ❿"
    assert py_iter_unicode(text) == [25105, 26377, 21966, 40, 49, 48, 41], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    
    
    text = "[] <s>"
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_CLS], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] </s>"
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_SEP], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <mask>"
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_MASK], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <pad>"
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_PAD], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <unk>"
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_UNK], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <nl>"
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_NL], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    
    
    text = "[] <s> "
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_CLS], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] </s> "
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_SEP], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <mask> "
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_MASK], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <pad> "
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_PAD], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <unk> "
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_UNK], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "[] <nl> "
    assert py_iter_unicode(text) == [91, 93, SPECIAL_TOKEN_NL   ], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    
    
    text = "<s> "
    assert py_iter_unicode(text) == [SPECIAL_TOKEN_CLS], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "</s> "
    assert py_iter_unicode(text) == [SPECIAL_TOKEN_SEP], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "<mask> "
    assert py_iter_unicode(text) == [SPECIAL_TOKEN_MASK], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "<pad> "
    assert py_iter_unicode(text) == [SPECIAL_TOKEN_PAD], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "<unk> "
    assert py_iter_unicode(text) == [SPECIAL_TOKEN_UNK], "Failed for `%s` %r"%(text, py_iter_unicode(text))
    
    text = "<nl> "
    assert py_iter_unicode(text) == [SPECIAL_TOKEN_NL], "Failed for `%s` %r"%(text, py_iter_unicode(text))
test()


















single_words_only = set()
single_words_any  = set()
irregular_exceptions = {}



prefixes = set()

# hashmap<str,hashset>
suffixes = {}
roots = set()

all_parts = {}
cdef int part_id = 0
all_parts_list = []

all_vocabs = {}
cdef int vocab_id = 0
all_vocabs_list = []


part_id_to_vocab_id = {}

no_prefixed_by_of = {}
only_suffixed_by_of = {}
no_suffixed_by_of = {}
no_suffixed_by_of_roots = {}
no_suffixed_by_of_nonroots = {}


cdef unordered_map[int, unordered_set[int]] true_no_prefixed_by_of = unordered_map[int, unordered_set[int]]()
cdef unordered_map[int, unordered_set[int]] true_only_suffixed_by_of = unordered_map[int, unordered_set[int]]()
cdef unordered_map[int, unordered_set[int]] true_no_suffixed_by_of = unordered_map[int, unordered_set[int]]()
cdef unordered_map[int, unordered_set[int]] true_no_suffixed_by_of_roots = unordered_map[int, unordered_set[int]]()
cdef unordered_map[int, unordered_set[int]] true_no_suffixed_by_of_nonroots = unordered_map[int, unordered_set[int]]()




from time import sleep

cdef bint adjacent_violate_c(int A_part, int B_part) nogil:
    cdef int A = A_part / 9
    cdef int A_part_merge_mode = (A_part - A * 9) / 3
    cdef int A_part_form = A_part % 3 

    cdef int B = B_part / 9
    cdef int B_part_merge_mode = (B_part - B * 9) / 3
    cdef int B_part_form = B_part % 3 

    cdef bint A_is_root = A_part_form == ROOT

    if (
        (A_part_merge_mode == MERGE_MODE_SINGLE and B_part_merge_mode != MERGE_MODE_SINGLE) or 
        A_part_merge_mode != MERGE_MODE_SINGLE and B_part_merge_mode == MERGE_MODE_SINGLE
        ):
        return False

    #if B in no_prefixed_by_of and A in no_prefixed_by_of[B]:
    if (true_no_prefixed_by_of.find(B) != true_no_prefixed_by_of.end() and 
        true_no_prefixed_by_of[B].find(A) != true_no_prefixed_by_of[B].end()):
        return True

        
    #if A in only_suffixed_by_of and B not in only_suffixed_by_of[A]:
    if (true_only_suffixed_by_of.find(A) != true_only_suffixed_by_of.end() and 
        true_only_suffixed_by_of[A].find(B) == true_only_suffixed_by_of[A].end()):
        return True
        
    if A_is_root:
        #if A in no_suffixed_by_of_roots and (B in no_suffixed_by_of_roots[A] or len(no_suffixed_by_of_roots[A]) == 0):
        if (true_no_suffixed_by_of_roots.find(A) != true_no_suffixed_by_of_roots.end() and 
            (
                true_no_suffixed_by_of_roots[A].size() == 0 or 
                true_no_suffixed_by_of_roots[A].find(B) != true_no_suffixed_by_of_roots[A].end()
            )):
            return True

    #elif A in no_suffixed_by_of_nonroots and (B in no_suffixed_by_of_nonroots[A] or len(no_suffixed_by_of_nonroots[A]) == 0):
    elif (
        true_no_suffixed_by_of_nonroots.find(A) != true_no_suffixed_by_of_nonroots.end() and 
            (
                true_no_suffixed_by_of_nonroots[A].size() == 0 or 
                true_no_suffixed_by_of_nonroots[A].find(B) != true_no_suffixed_by_of_nonroots[A].end()
            )
    ):
        return True
        
    #if A in no_suffixed_by_of and B in no_suffixed_by_of[A]:
    if (true_no_suffixed_by_of.find(A) != true_no_suffixed_by_of.end() and 
        true_no_suffixed_by_of[A].find(B) != true_no_suffixed_by_of[A].end()):
        return True

    return False
    
    
cdef bint adjacent_violate_str(str A, str B, bint A_is_root):
    if B in no_prefixed_by_of and A in no_prefixed_by_of[B]:
        return True
    if A in only_suffixed_by_of and B not in only_suffixed_by_of[A]:
        return True
    if A_is_root:
        if A in no_suffixed_by_of_roots and (B in no_suffixed_by_of_roots[A] or len(no_suffixed_by_of_roots[A]) == 0):
            return True
    elif A in no_suffixed_by_of_nonroots and (B in no_suffixed_by_of_nonroots[A] or len(no_suffixed_by_of_nonroots[A]) == 0):
        return True
    if A in no_suffixed_by_of and B in no_suffixed_by_of[A]:
        return True
    return False


                
cdef extern from "Python.h":
    const char* PyUnicode_AsUTF8(object unicode)
    const char* PyUnicode_AsUTF8AndSize(object unicode, Py_ssize_t *size)

    
cdef struct AB:
    int A
    int B
    int A_merge_mode
    int B_merge_mode
    char* morphed
    int longest_length
    int B_length
    int morphed_length
    long key
    float score

cdef struct ABC:
    int A
    int B
    int C
    int A_merge_mode
    int B_merge_mode
    int C_merge_mode
    char* morphed
    int longest_length
    int morphed_length
    long key
    float score
    

cdef vector[char*] stay_alive

cdef long hash_string(str s):
    return hash(s)


cdef vector[AB] morphed_suffixes_merge_suffixes
cdef vector[AB] morphed_roots_merge_suffixes



cdef vector[AB] get_morphed_suffixes_merge_suffixes(int min_left = 1):
    cdef int len_s, len_p, len_new_morphed, len_A, len_B
    cdef char* temp_char_star
    cdef char* orig_char_star
    cdef Py_ssize_t converted_size
    cdef float score
    morphed_suffixes_merge_suffixes.reserve(65536)

    for suffix, prefixers in suffixes.items():
        for p in prefixers:
            if not p:
                continue
            for s in suffixes:
                if s not in all_parts: # or s in irregular_exceptions:
                    continue
                if s.endswith(p):
                    A = s
                    B = suffix
                    if A == B:
                        continue
                        
                    if (
                        (A in single_words_only and (B not in single_words_any and B not in single_words_only)) or 
                        (B in single_words_only and (A not in single_words_any and A not in single_words_only))
                        ):
                        continue
                    if adjacent_violate_str(A, B, False):
                        continue
                        

                    A_merge_mode = (MERGE_MODE_BOTH if A in single_words_any else 
                        (MERGE_MODE_SINGLE if A in single_words_only else 
                        MERGE_MODE_NORMAL))
                    B_merge_mode = (MERGE_MODE_BOTH if B in single_words_any else 
                        (MERGE_MODE_SINGLE if B in single_words_only else 
                        MERGE_MODE_NORMAL))

                    len_s = len_A = len(s)
                    len_B = len(B)
                    len_p = len(p)
                        
                    if len_s - len_p < min_left:
                        continue
                        
                    overlapped = s[len_s-len_p:]
                    
                    if overlapped == 'e' or overlapped == 'o' or overlapped == 'y':
                        score = 0
                    else:
                        score = -len_p
                    
                    new_morphed = s[:len_s-len_p]+suffix
                    len_new_morphed = len(new_morphed)

                    orig_char_star = PyUnicode_AsUTF8AndSize(new_morphed, &converted_size)
                    temp_char_star = <char *>malloc(len_new_morphed * sizeof(char))
                    memcpy(temp_char_star, orig_char_star, converted_size + 1)

                    stay_alive.push_back(temp_char_star)
                    morphed_suffixes_merge_suffixes.push_back( 
                        AB(
                            all_parts[A],
                            all_parts[B], 
                            A_merge_mode,
                            B_merge_mode,
                            temp_char_star, 
                            len_B if len_B > len_A else len_A, 
                            len_B, 
                            len_new_morphed, 
                            hash_string(new_morphed), 
                            score 
                        )
                    )
                    assert morphed_suffixes_merge_suffixes[morphed_suffixes_merge_suffixes.size()-1].morphed == temp_char_star
                

cdef vector[AB] get_morphed_roots_merge_suffixes(int min_left = 2):
    cdef int len_s, len_p, len_new_morphed, len_A, len_B
    cdef float score
    cdef char* temp_char_star
    cdef char* orig_char_star
    cdef Py_ssize_t converted_size
    morphed_roots_merge_suffixes.reserve(1048576)
    
    for suffix, prefixers in suffixes.items():
        for p in prefixers:
            if not p:
                continue
            for s in roots:
                if s not in all_parts or s in irregular_exceptions:
                    continue
                if s.endswith(p):
                    A = s
                    B = suffix
                        
                    if (
                        (A in single_words_only and (B not in single_words_any and B not in single_words_only)) or 
                        (B in single_words_only and (A not in single_words_any and A not in single_words_only))
                        ):
                        continue
                    if adjacent_violate_str(A, B, True):
                        continue
                        
                    A_merge_mode = (MERGE_MODE_BOTH if A in single_words_any else 
                        (MERGE_MODE_SINGLE if A in single_words_only else 
                        MERGE_MODE_NORMAL))
                    B_merge_mode = (MERGE_MODE_BOTH if B in single_words_any else 
                        (MERGE_MODE_SINGLE if B in single_words_only else 
                        MERGE_MODE_NORMAL))

                    len_s = len_A = len(s)
                    len_B = len(B)
                    len_p = len(p)
                        
                    if len_s - len_p < min_left:
                        continue

                    overlapped = s[len_s-len_p:]
                    
                    if overlapped == 'e' or overlapped == 'o' or overlapped == 'y':
                        score = 0
                    else:
                        score = -len_p
                        
                    new_morphed = s[:len_s-len_p]+suffix
                    len_new_morphed = len(new_morphed)
                    
                    orig_char_star = PyUnicode_AsUTF8AndSize(new_morphed, &converted_size)
                    temp_char_star = <char *>malloc(len_new_morphed * sizeof(char))
                    memcpy(temp_char_star, orig_char_star, converted_size + 1)

                    stay_alive.push_back(temp_char_star)
                    morphed_roots_merge_suffixes.push_back( 
                        AB(
                            all_parts[A],
                            all_parts[B], 
                            A_merge_mode,
                            B_merge_mode,
                            temp_char_star, 
                            len_B if len_B > len_A else len_A, 
                            len_B, 
                            len_new_morphed, 
                            hash_string(new_morphed), 
                            score 
                        )
                    )
                    assert morphed_roots_merge_suffixes[morphed_roots_merge_suffixes.size()-1].morphed == temp_char_star



cdef class ABCResult:
    cdef ABC content
    def __init__(self, ABC content):
        self.content = content




def mix_suffixes_suffixes():
    cdef int min_left = 1
    cdef AB ab, bc
    cdef int A, B, C, D, morphed_length, B_length, len_new_morphed
    cdef vector[AB] bucket
    cdef long key
    cdef char* temp_char_star
    cdef char* orig_char_star
    cdef Py_ssize_t converted_size
    cdef unordered_map[int, vector[AB]] temp_index

    for i in range(<int>morphed_suffixes_merge_suffixes.size()):
        ab = morphed_suffixes_merge_suffixes[i]
        A = ab.A
        B = ab.B
        if ab.morphed_length - ab.B_length < min_left:
            continue

        if temp_index.find(A) == temp_index.end():
            temp_index[A] = vector[AB]()
            
        temp_index[A].push_back(ab)


    for i in range(<int>morphed_suffixes_merge_suffixes.size()):
        ab = morphed_suffixes_merge_suffixes[i]
        A = ab.A
        B = ab.B
        morphed_length = ab.morphed_length
        B_length = ab.B_length
        
        A_merge_mode = ab.A_merge_mode
        B_merge_mode = ab.B_merge_mode
        
        if temp_index.find(B) != temp_index.end():
            bucket = temp_index[B]
            for j in range(<int>bucket.size()):
                bc = bucket[j]
                C = bc.A
                D = bc.B

                C_merge_mode = bc.B_merge_mode

                assert morphed_length > B_length

                assert ab.morphed != NULL
                assert bc.morphed != NULL

                new_morphed = (
                    (<bytes>ab.morphed).decode("utf-8")[:morphed_length-B_length] + 
                    (<bytes>bc.morphed).decode("utf-8")
                    )
                key = hash_string(new_morphed)

                prefixes_lengths[key] = morphed_length + bc.morphed_length - B_length

                len_new_morphed = len(new_morphed)
                orig_char_star = PyUnicode_AsUTF8AndSize(new_morphed, &converted_size)
                temp_char_star = <char *>malloc(len_new_morphed * sizeof(char))
                memcpy(temp_char_star, orig_char_star, converted_size + 1)

                stay_alive.push_back(temp_char_star)
                yield ABCResult(
                    ABC(
                        A,
                        B,
                        D, 
                        A_merge_mode,
                        B_merge_mode,
                        C_merge_mode, 
                        temp_char_star, 
                        ab.longest_length if ab.longest_length > bc.longest_length else bc.longest_length,
                        len_new_morphed,
                        key, 
                        ab.score + bc.score 
                    )
                )

                
def mix_roots_suffixes():
    cdef int min_left = 2
    cdef AB ab, bc
    cdef int A, B, C, D, morphed_length, B_length, len_new_morphed
    cdef vector[AB] bucket
    cdef long key
    cdef char* temp_char_star
    cdef char* orig_char_star
    cdef Py_ssize_t converted_size
    cdef unordered_map[int, vector[AB]] temp_index

    for i in range(<int>morphed_roots_merge_suffixes.size()):
        ab = morphed_roots_merge_suffixes[i]
        A = ab.A
        B = ab.B
        if ab.morphed_length - ab.B_length < min_left:
            continue

        if temp_index.find(A) == temp_index.end():
            temp_index[A] = vector[AB]()
            
        temp_index[A].push_back(ab)


    for i in range(<int>morphed_roots_merge_suffixes.size()):
        ab = morphed_roots_merge_suffixes[i]
        A = ab.A
        B = ab.B
        morphed_length = ab.morphed_length
        B_length = ab.B_length
        
        A_merge_mode = ab.A_merge_mode
        B_merge_mode = ab.B_merge_mode
            
        if temp_index.find(B) != temp_index.end():
            bucket = temp_index[B]
            for j in range(<int>bucket.size()):
                bc = bucket[j]
                C = bc.A
                D = bc.B

                C_merge_mode = bc.B_merge_mode

                assert morphed_length > B_length

                assert ab.morphed != NULL
                assert bc.morphed != NULL

                new_morphed = (
                    (<bytes>ab.morphed).decode("utf-8")[:morphed_length-B_length] + 
                    (<bytes>bc.morphed).decode("utf-8")
                    )
                key = hash_string(new_morphed)

                prefixes_lengths[key] = morphed_length + bc.morphed_length - B_length

                len_new_morphed = len(new_morphed)
                orig_char_star = PyUnicode_AsUTF8AndSize(new_morphed, &converted_size)
                temp_char_star = <char *>malloc(len_new_morphed * sizeof(char))
                memcpy(temp_char_star, orig_char_star, converted_size + 1)

                stay_alive.push_back(temp_char_star)
                yield ABCResult(
                    ABC(
                        A,
                        B,
                        D, 
                        A_merge_mode,
                        B_merge_mode,
                        C_merge_mode, 
                        temp_char_star, 
                        ab.longest_length if ab.longest_length > bc.longest_length else bc.longest_length,
                        len_new_morphed,
                        key, 
                        ab.score + bc.score 
                    )
                )

                
                

for e in (
    no_prefixed_by_of,
    only_suffixed_by_of,
    no_suffixed_by_of_roots,
    no_suffixed_by_of_nonroots,
    no_suffixed_by_of
):
    for k, v in list(e.items()):
        if k not in prefixes and k not in roots and k not in suffixes:
            print('no', k)
            del e[k]
        for c in list(v):
            if not c:
                v.remove(c)
                continue
            if c not in prefixes and c not in roots and c not in suffixes:
                print('not found:', c)
                v.remove(c)
                
                
                
                

















cdef void ensure_bucket(long key):
    global trie_values
    cdef vector[Parts] bucket
    if trie_values.find(key) == trie_values.end():
        bucket = vector[Parts]()
        bucket.reserve(8)
        trie_values[key] = bucket
        

cdef int get_part(int content, int form, int merge_mode) nogil:
    return (content*9) + (merge_mode*3) + (form)


cdef unordered_map[long, vector[Parts]] trie_values
cdef unordered_map[long, int] prefixes_lengths
cdef unordered_map[long, int] prefixes_max_part_lengths

cdef unordered_map[int, int] true_part_id_to_vocab_id

cdef unordered_map[int, vector[Parts]] true_trie_values
cdef unordered_map[int, Parts] part_id_to_irregular_parts
cdef unordered_map[int, int] true_prefixes_lengths

cdef unordered_map[char, int] true_all_alphabets

def convert_part_id_to_vocab_id(int x):
    if true_part_id_to_vocab_id.find(x) != true_part_id_to_vocab_id.end():
        return true_part_id_to_vocab_id[x]
    return -1
def convert_str_to_trie_id(str x):
    return trie_obj[x]


def convert_trie_id_to_parts(int x):
    return true_trie_values[x]

    

def gen(bint debug=False):
    global mix_two_morphed_h
    global mix_two_morphed_k
    global part_id
    cdef vector[Parts]* bucket
    cdef vector[int] vec_part
    cdef int k, n, i,
    cdef long key, kk

    print('Loading prefixes')
    for e in prefixes:
        k = all_parts[e]
        key = hash_string(e)
        ensure_bucket(key)
        #print('%-20d: %s'%(key, e))
        bucket = &trie_values[key]
        vec_part = vector[int]()
        vec_part.reserve(1)
        kk = get_part(k, 
                    PREFIX, 
                    MERGE_MODE_BOTH if e in single_words_any else 
                    (MERGE_MODE_SINGLE if e in single_words_only else 
                    MERGE_MODE_NORMAL)
                    )
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "%s (PREFIX, %s) not found in part_id_to_vocab_id"%(e,         
                    'MERGE_MODE_BOTH' if e in single_words_any else 
                    ('MERGE_MODE_SINGLE' if e in single_words_only else 
                    'MERGE_MODE_NORMAL'))

        if debug:
            print('gen >> prefix :: '+e+' [part_id:%s] '%kk )
        vec_part.push_back(kk)
        bucket.push_back((vec_part, 0, 0, len(e)))
        prefixes_lengths[key] = prefixes_max_part_lengths[key] = len(e)
        
        assert len(e) > 0
        assert trie_values[key].size() > 0


        
        yield e.encode('utf-8')

    print('Loading roots')
    for e in roots:
        k = all_parts[e]
        key = hash_string(e)
        ensure_bucket(key)
        bucket = &trie_values[key]
        vec_part = vector[int]()
        vec_part.reserve(1)
        kk = get_part(k, 
                    ROOT, 
                    MERGE_MODE_BOTH if e in single_words_any else 
                    (MERGE_MODE_SINGLE if e in single_words_only else 
                    MERGE_MODE_NORMAL)
                    )
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "%s: %s (ROOT, %s) not found in part_id_to_vocab_id"%(kk, e,         
                    'MERGE_MODE_BOTH' if e in single_words_any else 
                    ('MERGE_MODE_SINGLE' if e in single_words_only else 
                    'MERGE_MODE_NORMAL'))
        
        if debug:
            print('gen >> roots  :: '+e+' [part_id:%s] '%kk )
        vec_part.push_back(kk)
        bucket.push_back((vec_part, -ROOT_PUNISHMENT, 0, len(e)))
        prefixes_lengths[key] = prefixes_max_part_lengths[key] = len(e)
        assert len(e) > 0
        assert trie_values[key].size() > 0
        yield e.encode('utf-8')

    print('Loading suffixes')
    for e in suffixes:
        k = all_parts[e]
        key = hash_string(e)
        ensure_bucket(key)
        bucket = &trie_values[key]
        vec_part = vector[int]()
        vec_part.reserve(1)
        
        kk = get_part(k, 
                    SUFFIX, 
                    MERGE_MODE_BOTH if e in single_words_any else 
                    (MERGE_MODE_SINGLE if e in single_words_only else 
                    MERGE_MODE_NORMAL)
                    )
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "%s (SUFFIX, %s) not found in part_id_to_vocab_id"%(e,         
                    'MERGE_MODE_BOTH' if e in single_words_any else 
                    ('MERGE_MODE_SINGLE' if e in single_words_only else 
                    'MERGE_MODE_NORMAL'))
        
        if debug:
            print('gen >> suffix  :: '+e+' [part_id:%s] '%kk )
        vec_part.push_back(kk)
        bucket.push_back((vec_part, 0, 0, len(e)))
        prefixes_lengths[key] = prefixes_max_part_lengths[key] = len(e)
        assert len(e) > 0
        assert trie_values[key].size() > 0
        yield e.encode('utf-8')
        

    print('Loading irregular_exceptions')
    for e in irregular_exceptions:
        #print('irregular_exceptions: %s'%e)
        if e in all_parts:
            continue

        part_id += 1
        k = all_parts[e] = part_id
        all_parts_list.append(e)
        key = hash_string(e)
        ensure_bucket(key)
        bucket = &trie_values[key]
        vec_part = vector[int]()
        vec_part.reserve(1)
        kk = get_part(k, 
                    ROOT, 
                    MERGE_MODE_BOTH if e in single_words_any else 
                    (MERGE_MODE_SINGLE if e in single_words_only else 
                    MERGE_MODE_NORMAL)
                    )
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "%s: %s (ROOT, %s) not found in part_id_to_vocab_id"%(kk, e,         
                    'MERGE_MODE_BOTH' if e in single_words_any else 
                    ('MERGE_MODE_SINGLE' if e in single_words_only else 
                    'MERGE_MODE_NORMAL'))
        
        if debug:
            print('gen >> roots  :: '+e+' [part_id:%s] '%kk )
        vec_part.push_back(kk)
        bucket.push_back((vec_part, -ROOT_PUNISHMENT, 0, len(e)))
        prefixes_lengths[key] = prefixes_max_part_lengths[key] = len(e)
        assert len(e) > 0
        assert trie_values[key].size() > 0
        yield e.encode('utf-8')
            


    print('Loading morphed_suffixes_merge_suffixes')
    get_morphed_suffixes_merge_suffixes()
    print('Loaded morphed_roots_merge_suffixes (%d)'%morphed_suffixes_merge_suffixes.size())

    print('Loading morphed_roots_merge_suffixes')
    get_morphed_roots_merge_suffixes()
    print('Loaded morphed_roots_merge_suffixes (%d)'%morphed_roots_merge_suffixes.size())
    
    cdef AB ret
    cdef ABC ret2

    print('Processing morphed_suffixes_merge_suffixes')
    for i in range(<int>morphed_suffixes_merge_suffixes.size()):
        ret = morphed_suffixes_merge_suffixes[i]
        ensure_bucket(ret.key)
        bucket = &trie_values[ret.key]
        
        vec_part = vector[int]()
        vec_part.reserve(2)
        kk = get_part(ret.A, SUFFIX, ret.A_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "SS AB 1: %r"%ret
        kk = get_part(ret.B, SUFFIX, ret.B_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "SS AB 2: %r"%ret

        vec_part.push_back(get_part(ret.A, SUFFIX, ret.A_merge_mode))
        vec_part.push_back(get_part(ret.B, SUFFIX, ret.B_merge_mode))
        bucket.push_back((vec_part, ret.score - 1, 0, ret.longest_length))
        prefixes_max_part_lengths[ret.key] = ret.longest_length
        prefixes_lengths[ret.key] = ret.morphed_length
        yield (<bytes>ret.morphed)
        
    print('Processing morphed_roots_merge_suffixes')
    for i in range(<int>morphed_roots_merge_suffixes.size()):
        ret = morphed_roots_merge_suffixes[i]
        ensure_bucket(ret.key)
        bucket = &trie_values[ret.key]

        kk = get_part(ret.A, ROOT, ret.A_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "RS AB 1: %r"%ret
        kk = get_part(ret.B, SUFFIX, ret.B_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "RS AB 2: %r"%ret
        
        vec_part = vector[int]()
        vec_part.reserve(2)
        vec_part.push_back(get_part(ret.A, ROOT, ret.A_merge_mode))
        vec_part.push_back(get_part(ret.B, SUFFIX, ret.B_merge_mode))
        bucket.push_back((vec_part, ret.score-ROOT_PUNISHMENT - 1, 0, ret.longest_length))
        
        prefixes_max_part_lengths[ret.key] = ret.longest_length
        prefixes_lengths[ret.key] = ret.morphed_length
        yield (<bytes>ret.morphed)

    cdef ABCResult ABC_result
    print('Loading morphed_suffixes_merge_suffixes mix morphed_suffixes_merge_suffixes')
    
    for ABC_result in mix_suffixes_suffixes():
        ret2 = ABC_result.content
        ensure_bucket(ret2.key)
        bucket = &trie_values[ret2.key]

        
        kk = get_part(ret2.A, SUFFIX, ret2.A_merge_mode)
        if debug: assert kk in part_id_to_vocab_id, "SSS ABC 1: %r"%ret2
        kk = get_part(ret2.B, SUFFIX, ret2.B_merge_mode)
        if debug: assert kk in part_id_to_vocab_id, "SSS ABC 2: %r"%ret2
        kk = get_part(ret2.C, SUFFIX, ret2.C_merge_mode)
        if debug: assert kk in part_id_to_vocab_id, "SSS ABC 3: %r"%ret2


        vec_part = vector[int]()
        vec_part.reserve(3)
        vec_part.push_back(get_part(ret2.A, SUFFIX, ret2.A_merge_mode))
        vec_part.push_back(get_part(ret2.B, SUFFIX, ret2.B_merge_mode))
        vec_part.push_back(get_part(ret2.C, SUFFIX, ret2.C_merge_mode))
        bucket.push_back((vec_part, ret2.score - 2, 0, ret2.longest_length))

        prefixes_max_part_lengths[ret2.key] = ret2.longest_length
        prefixes_lengths[ret2.key] = ret2.morphed_length
        yield (<bytes>ret2.morphed)
        free(stay_alive[stay_alive.size()-1])
        stay_alive.pop_back()
        
    print('Loading morphed_roots_merge_suffixes mix morphed_suffixes_merge_suffixes')

    for ABC_result in mix_roots_suffixes():

        ret2 = ABC_result.content
        ensure_bucket(ret2.key)
        bucket = &trie_values[ret2.key]

        kk = get_part(ret2.A, ROOT, ret2.A_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "RSS ABC 1: %r"%ret2
        kk = get_part(ret2.B, SUFFIX, ret2.B_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "RSS ABC 2: %r"%ret2
        kk = get_part(ret2.C, SUFFIX, ret2.C_merge_mode)
        if debug: assert kk in part_id_to_vocab_id or e in irregular_exceptions, "RSS ABC 3: %r"%ret2


        vec_part = vector[int]()
        vec_part.reserve(3)
        vec_part.push_back(get_part(ret2.A, ROOT, ret2.A_merge_mode))
        vec_part.push_back(get_part(ret2.B, SUFFIX, ret2.B_merge_mode))
        vec_part.push_back(get_part(ret2.C, SUFFIX, ret2.C_merge_mode))
        bucket.push_back((vec_part, ret2.score-ROOT_PUNISHMENT - 2, 0, ret2.longest_length))
        
        prefixes_max_part_lengths[ret2.key] = ret2.longest_length
        prefixes_lengths[ret2.key] = ret2.morphed_length
        yield (<bytes>ret2.morphed)
        free(stay_alive[stay_alive.size()-1])
        stay_alive.pop_back()


from cython.operator cimport dereference as deref, preincrement as inc

all_alphabets = ""
def load_data(alphabets = "abcdefghijklmnopqrstuvwxyz", debug=False):
    global vocab_id, part_id, all_alphabets
    all_alphabets = alphabets
    
    fn = get_file('words2.txt')
    with open(fn) as f:
        words = f.read()
        
    for e in alphabets:
        vocab_id += 1
        part_id += 1
        all_parts[e] = part_id
        all_parts_list.append(e)
        all_vocabs[e] = vocab_id - 1
        all_vocabs_list.append(e)

        single_words_any.add(e)

        part_id_to_vocab_id[
            get_part(
                part_id, 
                SUFFIX, 
                MERGE_MODE_BOTH
                )] = vocab_id - 1

    '''
    must_have_prefixes = "-"
    for e in must_have_prefixes:
        vocab_id += 1
        part_id += 1
        all_parts[e] = part_id
        all_parts_list.append(e)
        all_vocabs[e] = vocab_id - 1
        all_vocabs_list.append(e)
        prefixes.add(e)
        suffixes[e] = set([''])

        single_words_any.add(e)

        part_id_to_vocab_id[
            get_part(
                part_id, 
                PREFIX, 
                MERGE_MODE_BOTH
                )] = vocab_id - 1
        part_id_to_vocab_id[
            get_part(
                part_id, 
                SUFFIX, 
                MERGE_MODE_BOTH
                )] = vocab_id - 1
    '''

    with open(get_file('single_words.txt')) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('*'):
                single_words_any.add(line[1:])
                continue
                
            if not line:
                continue

            single_words_only.add(line)
    with open(get_file('irregular_exceptions.json')) as f:
        irregular_exceptions.update(json.load(f))


    with open(get_file('words2.txt')) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            splitted = ['','','']
            for i, e in enumerate(line.split('|')):
                splitted[i] = e
                
            a,b,c = splitted
            
            token_id = 0
            
            for e in a.split(','):
                if e in irregular_exceptions:
                    if len(a.split(',')) != 1:
                        raise
                    # this will be decoded like normal token, but decomposed later
                    # so no token_id
                    token_id = -1
                    break
                if e in all_vocabs:
                    token_id = all_vocabs[e]
                    
            if token_id == 0:
                vocab_id += 1
                token_id = vocab_id
            
            index = 0
            for e in a.split(','):
                assert '+' not in e
                if not e:
                    continue
                E = e.strip('-')

                is_prefix = e.endswith('-')
                is_suffix = e.startswith('-')

                index += 1

                
                
                    
                if E not in all_parts:
                    part_id += 1
                    all_parts[E] = part_id
                    all_parts_list.append(E)
                    
                    
                if is_suffix:
                    if E in single_words_only:
                        single_words_any.add(E)
                        single_words_only.remove(E)
                    if E not in suffixes:
                        suffixes[E] = set()
                        
                        if not c:
                            suffixes[E].add('')
                    if b and index == 1:
                        for bb in b.split(','):
                            bb = bb.strip()
                            assert bb.startswith('+')
                            suffixes[E].add(bb[1:])
                elif is_prefix:
                    if E in single_words_only:
                        single_words_any.add(E)
                        single_words_only.remove(E)
                        
                    prefixes.add(E)
                else:
                    if e in single_words_only:
                        single_words_any.add(e)
                        single_words_only.remove(e)
                    roots.add(e)

                    
                if token_id > 0:
                    all_vocabs[e] = token_id
                    kk = get_part(
                            all_parts[E], 
                            PREFIX if is_prefix else (SUFFIX if is_suffix else ROOT), 
                            MERGE_MODE_BOTH if E in single_words_any else 
                            (MERGE_MODE_SINGLE if E in single_words_only else 
                            MERGE_MODE_NORMAL)
                          )
                    part_id_to_vocab_id[kk] = token_id - 1
                    if debug:
                        print('[part_id:'+str(kk)+']: '+e+'(' + ('PREFIX' if is_prefix else ('SUFFIX' if is_suffix else 'ROOT')) + ') '+  ('MERGE_MODE_BOTH' if E in single_words_any else 
                                    ('MERGE_MODE_SINGLE' if E in single_words_only else 
                                    'MERGE_MODE_NORMAL')) + ' >> vocab_id:'+ str(token_id) + ' [part_content:%s]'%all_parts[E])
    for e in single_words_only:
        assert '-' not in e
        assert e
        if e not in all_parts:
            part_id += 1
            all_parts[e] = part_id
            all_parts_list.append(e)
            if token_id > 0:
                kk = get_part(
                        part_id, 
                        ROOT, 
                        MERGE_MODE_BOTH if e in single_words_any else 
                        (MERGE_MODE_SINGLE if e in single_words_only else 
                        MERGE_MODE_NORMAL)
                        )
                part_id_to_vocab_id[kk] = token_id - 1
                if debug:
                    print('[part_id:'+str(kk)+']: '+e+'(ROOT) '+  ('MERGE_MODE_BOTH' if e in single_words_any else 
                                ('MERGE_MODE_SINGLE' if e in single_words_only else 
                                'MERGE_MODE_NORMAL')) + ' >> vocab_id:'+ str(token_id) + ' [part_content:%s]'%all_parts[e])
            if e not in roots:
                roots.add(e)
    for e in prefixes:
        assert e in all_parts
    for e in roots:
        assert e in all_parts
    for e in suffixes:
        assert e in all_parts
    rules = {}
    with open(get_file('rules.py')) as f:
        exec(f.read(), rules)

    # A no prefixed by B
    # all roots
    no_prefixed_by_of.update({
        k: {e.strip() for e in v if e.strip() in all_parts}
        for k, v in rules['expand_allowed_reverse'].items() if k in all_parts
    })

    # A only suffixed by B
    only_suffixed_by_of.update({k:{e.strip() for e in (v or set()) if e.strip() in all_parts} for k, v in rules['expand_allowed'].items() if k in all_parts})
    rules['expand_allowed'].clear()
        
    # A no suffixed by B
    no_suffixed_by_of.update({
        k: {e.strip() for e in v if e.strip() in all_parts}
        for k, v in rules['pairs_exceptions'].items() if k in all_parts
    })

    # A no suffixed by B (root)
    no_suffixed_by_of_roots.update({
        k[0]: {e.strip() for e in v.split(',') if e.strip() in all_parts}
        for k, v in rules['expand_exceptions'].items() if k[1] and k[0] in all_parts
    })

    # A no suffixed by B (non-root)
    no_suffixed_by_of_nonroots.update({
        k[0]: {e.strip() for e in v.split(',') if e.strip() in all_parts}
        for k, v in rules['expand_exceptions'].items() if not k[1] and k[0] in all_parts
    })
            
    rules['expand_exceptions'].clear()



def make_irregular():
    cdef:
        str part, k
        bytes b_part
        int trie_id, part_id, i, m
        vector[Parts] repl, repl_combined, repl_combined_temp
        int len_contents
        Parts *temp_parts
        Parts parts
        float score, new_score, cmp_max_score
        vector[int] contents, new_contents
        vector[int]* temp_contents
        bint has_root
        int A_part, B_part
        int temp_parts_length
        int A_part_form, B_part_form
        int len_p_part_max, len_part_max, len_v, part_index


    for k, v in irregular_exceptions.items():
        part_id = get_part(all_parts[k],
                    ROOT, 
                    MERGE_MODE_BOTH if k in single_words_any else 
                    (MERGE_MODE_SINGLE if k in single_words_only else 
                    MERGE_MODE_NORMAL)
                    )
        repl_combined = vector[Parts]()
        len_v = len(v)
        part_index = 0
        if not v:
            continue
        for part in v:  
            part_index += 1
            try:
                trie_id = trie_obj[part]
                assert trie_id < true_trie_values.size(), 'true_trie_values size ? %s'%((k, v),)
                repl = true_trie_values[trie_id]
            except:
                repl = vector[Parts]()
                b_part = part.encode('utf8')
                _tokenize_word_to_parts(<char*>b_part, len(b_part), &repl, MIN_SCORE_RATIO, False)
                
            if repl_combined.size() == 0:
                for i in range(repl.size()):
                    repl_combined.push_back(repl[i])
            else:
                repl_combined_temp = vector[Parts]()
                for i in range(repl_combined.size()):
                    contents = repl_combined[i][0]
                    score = repl_combined[i][1]
                    len_contents = contents.size()
                    len_part_max = repl_combined[i][3]
                    for m in range(repl.size()):
                        temp_parts = &(repl[m])
                        temp_contents = &(temp_parts[0][0])
                        temp_parts_length = temp_contents.size()
                        
                        len_p_part_max = len_part_max if len_part_max > temp_parts[0][3] else temp_parts[0][3]

                        new_contents = vector[int]()
                        new_contents.reserve(len_contents+temp_parts_length)
                        for j in range(len_contents):
                            new_contents.push_back(contents[j])
                        for j in range(temp_parts_length):
                            new_contents.push_back(temp_contents[0][j])
                        
                        new_score = score + temp_parts[0][1] - 1

                        if part_index == len_v:
                            # Prefer splits that have a root
                            # if no root, punish! 
                            has_root = False
                            for m in range(len_contents+temp_parts_length):
                                A_part = new_contents[m]
                                A_part_form = A_part % 3
                                if A_part_form == ROOT:
                                    has_root = True
                                    break

                            if not has_root:
                                new_score -= 1
                                
                            # ends with prefix and start with suffix is not good
                            for m from len_contents+temp_parts_length-1 >= m >= 0: 
                                A_part = new_contents[m]
                                A_part_form = A_part % 3
                                if A_part_form == PREFIX:
                                    new_score -= 1
                                    break
                                elif A_part_form == SUFFIX:
                                    continue
                                break

                            for m in range(len_contents+temp_parts_length):
                                A_part = new_contents[m]
                                A_part_form = A_part % 3
                                if A_part_form == SUFFIX:
                                    new_score -= 1
                                    break
                                elif A_part_form == PREFIX:
                                    continue
                                break
                              

                        repl_combined_temp.push_back(
                            (new_contents, 
                            new_score, 
                            temp_parts[0][2], 
                            len_p_part_max)) 


                repl_combined = repl_combined_temp
            
            
        i = 0
        cmp_max_score = -100
        assert repl_combined.size() > 0, 'repl_combined size ? %s'%((k, v),)

        for j in range(repl_combined.size()):
            score = <float>repl_combined[j][1] + (<float>repl_combined[j][3] / 1000)
            if score > cmp_max_score:
                cmp_max_score = score
                i = j
                
        part_id_to_irregular_parts[part_id] = repl_combined[i]



def true_trie_values_to_np():
    cdef unordered_map[int, vector[Parts]].iterator it = true_trie_values.begin()
    cdef unordered_map[int, vector[Parts]].iterator end = true_trie_values.end()
        
    cdef int key, i, j, size, size2
    cdef vector[Parts] vector_parts
    cdef Parts parts
    cdef int part
    cdef vector[np.int32_t] buffer
    buffer.reserve(2**30)

    buffer.push_back(true_trie_values.size())

    while it != end:
        key = deref(it).first
        vector_parts = deref(it).second
        size = vector_parts.size()

        assert size > 0, "% zero size (:1)"%key

        buffer.push_back(key)
        buffer.push_back(size)


        for i in range(size):
            parts = vector_parts[i]
            size2 = parts[0].size()
            assert size2 > 0, "% zero size (:2)"%key

            if size2 == 1 and parts[0][0]%3 == ROOT:
                assert parts[1] < 0
                

            buffer.push_back(<int>(parts[1]*100))
            buffer.push_back(parts[3])
            buffer.push_back(size2)

            for j in range(size2):
                part = parts[0][j]
                assert part in part_id_to_vocab_id or all_parts_list[int(part/9) - 1] in irregular_exceptions, "part not in part_id_to_vocab_id (%s) [%s]"%(all_parts_list[int(part/9) - 1], int(part) )

                buffer.push_back(part)

        inc(it)

    cdef unordered_map[int, Parts].iterator it1 = part_id_to_irregular_parts.begin()
    cdef unordered_map[int, Parts].iterator end1 = part_id_to_irregular_parts.end()

    buffer.push_back(part_id_to_irregular_parts.size())
    while it1 != end1:
        key = deref(it1).first
        parts = deref(it1).second

        buffer.push_back(key)

        size2 = parts[0].size()
        assert size2 > 0, "% zero size (:2)"%key

        if size2 == 1 and parts[0][0]%3 == ROOT:
            assert parts[1] < 0

        buffer.push_back(<int>(parts[1]*100))
        buffer.push_back(parts[3])
        buffer.push_back(size2)

        for j in range(size2):
            part = parts[0][j]
            assert part in part_id_to_vocab_id or all_parts_list[int(part/9) - 1] in irregular_exceptions, "part not in part_id_to_vocab_id (%s) [%s]"%(all_parts_list[int(part/9) - 1], int(part) )

            buffer.push_back(part)


        inc(it1)





    cdef unordered_map[int, vector[Parts]].iterator it2
    cdef unordered_map[int, vector[Parts]].iterator end2

    def save_unordered_map_int_to_unordered_set_int(d):
        buffer.push_back(len(d))
        for k, v in list(d.items()):
            if k: 
                buffer.push_back(all_parts[k])
                s = {all_parts[c] for c in v}
                buffer.push_back(len(s))
                for e in s:
                    buffer.push_back(e)

    save_unordered_map_int_to_unordered_set_int(no_prefixed_by_of)
    save_unordered_map_int_to_unordered_set_int(only_suffixed_by_of)
    save_unordered_map_int_to_unordered_set_int(no_suffixed_by_of_roots)
    save_unordered_map_int_to_unordered_set_int(no_suffixed_by_of_nonroots)
    save_unordered_map_int_to_unordered_set_int(no_suffixed_by_of)

    


    #true_no_prefixed_by_of[all_parts[k]] = {all_parts[c] for c in v}
    #true_only_suffixed_by_of[all_parts[k]] = {all_parts[c] for c in v}
    #true_no_suffixed_by_of_roots[all_parts[k]] = {all_parts[c] for c in v}
    #true_no_suffixed_by_of_nonroots[all_parts[k]] = {all_parts[c] for c in v}
    #true_no_suffixed_by_of[all_parts[k]] = {all_parts[c] for c in v}

    cdef unordered_map[int, int].iterator it3 = true_prefixes_lengths.begin()
    cdef unordered_map[int, int].iterator end3 = true_prefixes_lengths.end()
    cdef int trie_id
    cdef int length

    buffer.push_back(true_prefixes_lengths.size())

    while it3 != end3:
        trie_id = deref(it3).first
        length = deref(it3).second

        buffer.push_back(trie_id)
        buffer.push_back(length)

        inc(it3)



    cdef unordered_map[int, int].iterator it5 = true_part_id_to_vocab_id.begin()
    cdef unordered_map[int, int].iterator end5 = true_part_id_to_vocab_id.end()

    buffer.push_back(true_part_id_to_vocab_id.size())

    while it5 != end5:
        trie_id = deref(it5).first
        length = deref(it5).second

        buffer.push_back(trie_id)
        buffer.push_back(length)

        inc(it5)


        
    buffer.push_back(len(all_alphabets))
    for ch in all_alphabets:
        buffer.push_back(<int>ord(ch))


    cdef np.ndarray[np.int32_t] data = pynp.empty(buffer.size(), dtype=pynp.int32)
    for i in prange(data.shape[0], nogil=True):
        data[i] = buffer[i]

    return data

import zlib
cdef Trie trie_obj = Trie()

def generate_trie(str path, str name, bint debug=False):
    global stay_alive, trie_obj
    cdef int i 

    load_data(debug=debug) 

    cdef int trie_id
    cdef long key

    trie_obj = Trie(gen(debug=debug))
    
    for i in range(<int>stay_alive.size()):
        free(stay_alive[i])

    trie_obj.save(os.path.join(path, name+'.trie'))

    cdef unordered_map[int, int] temp_mapping

    cdef unordered_map[long, vector[Parts]].iterator it = trie_values.begin()
    cdef unordered_map[long, vector[Parts]].iterator end = trie_values.end()
    cdef int size
    cdef vector[Parts] val
    while it != end:
        key = deref(it).first
        val = deref(it).second
        assert val.size() > 0, "trie_values"
        assert trie_values[key].size() > 0, "trie_values!"

        inc(it)



    for e in trie_obj:
        key = hash_string(e.decode())
        trie_id = trie_obj[e.decode()]

        temp_mapping[key] = trie_id
        
        if trie_values.find(key) == trie_values.end():
            raise Exception('key `%s` not found: %r'%(key, e.decode()))
            
        assert trie_values[key].size() > 0, "trie_values 2" 
        true_trie_values[trie_id] = trie_values[key]
        assert true_trie_values[trie_id].size() > 0, "true_trie_values"

    cdef unordered_map[long, int].iterator it2 = prefixes_lengths.begin()
    cdef unordered_map[long, int].iterator end2 = prefixes_lengths.end()
    cdef int length
    while it2 != end2:
        key = deref(it2).first
        length = deref(it2).second
        true_prefixes_lengths[temp_mapping[key]] = length

        inc(it2)



    for key, v in part_id_to_vocab_id.items():
        true_part_id_to_vocab_id[key] = v

    make_irregular()

    data = true_trie_values_to_np()

    with open(os.path.join(path, name+'.vals'), 'wb') as f:
        f.write(zlib.compress(data.data.tobytes(), level=1))  # level 1 is enough

    with open(os.path.join(path, name+'.vocabs'), 'w') as f:
        f.write("\n".join(all_parts_list))
        
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

cdef double get_time():
    cdef timespec ts
    cdef double current
    clock_gettime(CLOCK_REALTIME, &ts)
    current = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    return current 

cimport openmp
cdef openmp.omp_lock_t lock
openmp.omp_init_lock(&lock)
cdef bint loaded = False





def load(str path, str name, bint profile=False, bint debug=False):
    global all_parts_list, loaded
    openmp.omp_set_lock(&lock)
    cdef unordered_set[int] mapping
    cdef vector[Parts] vector_parts
    cdef vector[int] vector_part
    cdef Parts parts
    cdef int part
    cdef np.ndarray[np.int32_t, ndim=1] data
    cdef np.int32_t[::1] view 
    cdef double t0, t1

    if profile:
        t0 = get_time()
    with open(os.path.join(path, name+'.vocabs')) as f:
        all_parts_list = f.read().split('\n')
    if profile:
        t1 = get_time()
        print('vocabs read in %.4f s'%(t1-t0))
        
        
    if profile:
        t0 = get_time()
    trie_obj.load(os.path.join(path, name+'.trie'))
    if profile:
        t1 = get_time()
        print('trie_obj loaded in %.4f s'%(t1-t0))
        
    if profile:
        t0 = get_time()
    with open(os.path.join(path, name+'.vals'), 'rb') as f:
        data = pynp.frombuffer(zlib.decompress(f.read()), dtype=pynp.int32)
    if profile:
        t1 = get_time()
        print('vals read in %.4f s'%(t1-t0))
        
        
    if profile:
        t0 = get_time()
    cdef np.int32_t v
    cdef int trie_id, i, j, size, size2, part_id
    cdef float score
    cdef long k = 1, max_length = data.shape[0]
    cdef int true_trie_values_size = data[0], len_part_max

    
    while k < max_length and <int>true_trie_values.size() < true_trie_values_size:
        trie_id = data[k]
        k += 1
        size = data[k]
        k += 1
    
        vector_parts = vector[Parts]()
        vector_parts.reserve(size)

        for i in range(size):
            score = (<float>data[k]) / 100; k += 1
            len_part_max = data[k]; k += 1
            size2 = data[k]; k += 1
            
            vector_part = vector[int]()
            vector_part.reserve(size2)
            for j in range(size2):
                vector_part.push_back(data[k]) 
                k += 1
                
            #if debug:
            #    print(' '.join((all_parts_list[e//9-1]+' (%s)'%(FORMS[e%3])   ) for e in vector_part) +  'score: %s'%( score))
            
            parts = (vector_part, score, 0, len_part_max)
        
            vector_parts.push_back(parts)
        
        true_trie_values[trie_id] = vector_parts
        
    assert true_trie_values_size == <int>true_trie_values.size(), "true_trie_values size mismatch. Got %s, should be %s"%(true_trie_values.size(), true_trie_values_size)
        
    print('true_trie_values:', true_trie_values.size() )
    
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        
        score = (<float>data[k]) / 100; k += 1
        len_part_max = data[k]; k += 1
        size2 = data[k]; k += 1
        
        vector_part = vector[int]()
        vector_part.reserve(size2)
        for j in range(size2):
            vector_part.push_back(data[k]) 
            k += 1
            
        
        parts = (vector_part, score, 0, len_part_max)
        part_id_to_irregular_parts[part_id] = parts

    assert size == <int>part_id_to_irregular_parts.size(), "part_id_to_irregular_parts size mismatch. Got %s, should be %s"%(part_id_to_irregular_parts.size(), size)

    print('part_id_to_irregular_parts:', part_id_to_irregular_parts.size() )
    # print(part_id_to_irregular_parts)




        
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        size2 = data[k]; k += 1
        mapping = unordered_set[int]()
        for j in range(size2):
            mapping.insert(data[k]); k += 1
            
        true_no_prefixed_by_of[part_id] = mapping
        
    assert size == <int>true_no_prefixed_by_of.size(), \
        "true_no_prefixed_by_of size mismatch. Got %s, should be %s"%(true_no_prefixed_by_of.size(), size)
    print('true_no_prefixed_by_of:', true_no_prefixed_by_of.size() )
        
        
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        size2 = data[k]; k += 1
        mapping = unordered_set[int]()
        for j in range(size2):
            mapping.insert(data[k]); k += 1
            
        true_only_suffixed_by_of[part_id] = mapping
    
    assert size == <int>true_only_suffixed_by_of.size(), \
        "true_only_suffixed_by_of size mismatch. Got %s, should be %s"%(true_only_suffixed_by_of.size(), size)
    print('true_only_suffixed_by_of:', true_only_suffixed_by_of.size() )
        
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        size2 = data[k]; k += 1
        mapping = unordered_set[int]()
        for j in range(size2):
            mapping.insert(data[k]); k += 1
            
        true_no_suffixed_by_of_roots[part_id] = mapping
    
    assert size == <int>true_no_suffixed_by_of_roots.size(), \
        "true_no_suffixed_by_of_roots size mismatch. Got %s, should be %s"%(true_no_suffixed_by_of_roots.size(), size)
    print('true_no_suffixed_by_of_roots:', true_no_suffixed_by_of_roots.size() )
        
        
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        size2 = data[k]; k += 1
        mapping = unordered_set[int]()
        for j in range(size2):
            mapping.insert(data[k]); k += 1
            
        true_no_suffixed_by_of_nonroots[part_id] = mapping
    
    assert size == <int>true_no_suffixed_by_of_nonroots.size(), \
        "true_no_suffixed_by_of_nonroots size mismatch. Got %s, should be %s"%(true_no_suffixed_by_of_nonroots.size(), size)
    print('true_no_suffixed_by_of_nonroots:', true_no_suffixed_by_of_nonroots.size() )
        
        
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        size2 = data[k]; k += 1
        mapping = unordered_set[int]()
        for j in range(size2):
            mapping.insert(data[k]); k += 1
            
        true_no_suffixed_by_of[part_id] = mapping
    
    assert size == <int>true_no_suffixed_by_of.size(), \
        "true_no_suffixed_by_of size mismatch. Got %s, should be %s"%(true_no_suffixed_by_of.size(), size)
    print('true_no_suffixed_by_of:', true_no_suffixed_by_of.size() )
    
    
    
    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        true_prefixes_lengths[part_id] = data[k]; k += 1
    
    assert size == <int>true_prefixes_lengths.size(), \
        "true_prefixes_lengths size mismatch. Got %s, should be %s"%(true_prefixes_lengths.size(), size)
    
    print('true_prefixes_lengths:', true_prefixes_lengths.size() )


    size = data[k]; k += 1
    for i in range(size):
        part_id = data[k]; k += 1
        true_part_id_to_vocab_id[part_id] = data[k]; k += 1
    
    assert size == <int>true_part_id_to_vocab_id.size(), \
        "true_part_id_to_vocab_id size mismatch. Got %s, should be %s"%(true_part_id_to_vocab_id.size(), size)
    
    print('true_part_id_to_vocab_id:', true_part_id_to_vocab_id.size() )

    size = data[k]; k += 1
    for i in range(size):
        true_all_alphabets[data[k]] = i+1; k += 1


    

    assert k == max_length, "File size not match"


    if profile:
        t1 = get_time()
        print('vals loaded in %.4f s'%(t1-t0))

    loaded = True

    openmp.omp_unset_lock(&lock)
        






































cdef float MIN_SCORE_RATIO = 1.25

cdef extern from *:
    ctypedef int int128 "__int128_t"

ctypedef vector[int] vector_int
ctypedef lru_cache[int128, vector_int*] lru_cache_int128_vector_int_star

cdef lru_cache_int128_vector_int_star* tokenize_lru_cache = new lru_cache_int128_vector_int_star(1024*1024)


cdef int128** int128_power = <int128**> malloc(sizeof(int128*) * 26)

for i in range(26):
    int128_power[i] = <int128*> malloc(sizeof(int128) * 27)
    for j in range(27):
        int128_power[i][j] = (j+1) * <int128>26**i


cdef vector[int]* tokenize_word_auto_c(char* chars, int length, bint use_cache, bint to_vocab_id) nogil:
    cdef int max_call_depth, i
    cdef unsigned long cache_key = 0
    cdef vector[int]* result_contents
    cdef size_t t
    if length <= 26:
        for i in range(length):
            t = chars[i]
            if t == 39:
                cache_key += int128_power[26][i]
            elif t >= 97 and t <= 122:
                cache_key += int128_power[t-97][i]


    openmp.omp_set_lock(&lock)
    if use_cache and tokenize_lru_cache.exists(cache_key):
        openmp.omp_unset_lock(&lock)
        return tokenize_lru_cache.get(cache_key)

    result_contents = new vector[int]()
    result_contents.reserve(8)
    tokenize_lru_cache.put(cache_key, result_contents)
    openmp.omp_unset_lock(&lock)
    
    max_call_depth = 4

    _tokenize_word(chars, length, result_contents, max_call_depth, MIN_SCORE_RATIO, False)

    if to_vocab_id:
        for i in range(result_contents[0].size()):
            if true_part_id_to_vocab_id.find(result_contents[0][i]) == true_part_id_to_vocab_id.end():
                return NULL
            result_contents[0][i] = true_part_id_to_vocab_id[result_contents[0][i]]
            
    return result_contents

def tokenize_word_auto(str word, bint use_cache=False, bint to_vocab_id=False):
    cdef:
        bytes b_word = word.encode('utf-8')
    return tokenize_word_auto_c(<char *> b_word, len(b_word), use_cache, to_vocab_id)[0]





def tokenize_word(str word, int max_call_depth = 3, float min_score_ratio = 1.5, bint return_candidates = False, bint debug=False, bint to_vocab_id=False):
    cdef:
        vector[Parts] results
        bytes b_word = word.encode('utf-8')
        int length = len(word)

    if max_call_depth == 0:
        max_call_depth = 4

    if max_call_depth < 2:
        max_call_depth = 2

    if return_candidates:
        return _tokenize_word_candidates(<char *>b_word, length, max_call_depth, min_score_ratio, debug)

    cdef vector[int] result_contents = vector[int]()
    result_contents.reserve(8)
    _tokenize_word(<char *>b_word, length, &result_contents, max_call_depth, min_score_ratio, debug)

    if to_vocab_id:
        for i in range(result_contents.size()):
            if true_part_id_to_vocab_id.find(result_contents[i]) == true_part_id_to_vocab_id.end():
                return 0
            result_contents[i] = true_part_id_to_vocab_id[result_contents[i]]
    return result_contents


cdef vector[Parts] _tokenize_word_candidates(char* chars, int length, int max_call_depth, float min_score_ratio, bint debug) nogil:
    cdef:
        int i = 0, j = 0
        int cursor = 0, size, result_contents_size
        vector[vector[Parts]] cache = vector[vector[Parts]]()
        vector[int] contents = vector[int](), result_contents = vector[int]()
        Parts parts
        int call_depth = 0
        float max_score = -100
        bint cont = True, first=True
        float cmp_max_score, score
        vector[Parts] results

    result_contents.reserve(8)
    
    for i in range(length+1):
        cache.push_back(vector[Parts]())
    

    while cont:
        parts = (contents, 0, 0, 0)
        results = vector[Parts]()
        results.reserve(1024)
        tokenize_inner(chars, cursor, length, parts, &cache, max_call_depth, &max_score, min_score_ratio, call_depth, &results)
        size = results.size()
        if size == 0:
            break

        cmp_max_score = -100
        i = 0
        for j in range(size):
            if results[j][2] == length:
                cont = False
                break
            score = <float>results[j][1] + (<float>results[j][3] / 1000) + results[j][0].size()
            if score > cmp_max_score:
                cmp_max_score = score
                i = j

        if not cont:
            break

        result_contents.insert(result_contents.end(),results[i][0].begin(),results[i][0].end() - 1)

        contents = vector[int]()
        contents.push_back(results[i][0][results[i][0].size() - 1])

        first = False

        cursor = results[i][2]


        if debug:
            with gil:
                g = list(results)
                g.sort(key=lambda x: x[1] + len(x[0]), reverse=True)
                for e in [(' '.join((all_parts_list[e//9-1]+' (%s)'%FORMS[e%3]   ) for e in a), b+d/1000+len(a), c) for a, b, c, d in g][:5]:
                    print(e)
                print('cursor: %s'%cursor)
                print(' ')

    
    result_contents_size = result_contents.size()
    for i in range(size):
        for j from result_contents_size-1 >= j >= 0: 
            results[i][0].insert(results[i][0].begin(),result_contents[j])


    return results


cdef Parts _tokenize_word_to_parts(char* chars, int length, vector[Parts]* results, float min_score_ratio, bint debug) nogil:
    cdef:
        int i = 0, j = 0
        int cursor = 0, size
        vector[vector[Parts]] cache = vector[vector[Parts]]()
        vector[int] contents = vector[int]()
        Parts parts
        int call_depth = 0
        float max_score = -100
        float cmp_max_score , score

    for i in range(length+1):
        cache.push_back(vector[Parts]())

    parts = (contents, 0, 0, 0)
    
    results.reserve(1024)
    tokenize_inner(chars, cursor, length, parts, &cache, 100, &max_score, min_score_ratio, call_depth, results)

    i = 0
    cmp_max_score = -100

    for j in range(results.size()):
        if results[0][j][2] == length:
            score = <float>results[0][j][1] + (<float>results[0][j][3] / 1000)
            if score > cmp_max_score:
                cmp_max_score = score
                i = j
    if results.size() == 0:
        with gil:
            print("%s failed to be tokenized"%chars)
        return parts
    return results[0][i]



cdef int _tokenize_word(char* chars, int length, vector[int]* result_contents, int max_call_depth, float min_score_ratio, bint debug) nogil:
    cdef:
        int i = 0, j = 0
        int cursor = 0, size
        vector[vector[Parts]] cache = vector[vector[Parts]]()
        vector[int] contents = vector[int]()
        Parts parts
        int call_depth = 0
        float max_score = -100
        bint cont = True
        float cmp_max_score , score
        vector[Parts] results


    for i in range(length+1):
        cache.push_back(vector[Parts]())
    

    while cont:
        parts = (contents, 0, 0, 0)
        results = vector[Parts]()
        results.reserve(1024)
        tokenize_inner(chars, cursor, length, parts, &cache, max_call_depth, &max_score, min_score_ratio, call_depth, &results)
            


        size = results.size()
        if size == 0:
            return 0

        cmp_max_score = -100
        i = 0
        for j in range(size):
            if results[j][2] == length:
                cont = False
                break
            score = <float>results[j][1] + (<float>results[j][3] / 1000) + results[j][0].size()
            if score > cmp_max_score:
                cmp_max_score = score
                i = j

        if not cont:
            break

        result_contents.insert(result_contents.end(),results[i][0].begin(),results[i][0].end() - 1)

        contents = vector[int]()
        contents.push_back(results[i][0][results[i][0].size() - 1])
        
        cursor = results[i][2]


    i = -1
    cmp_max_score = -100

    for j in range(size):
        if results[j][2] == length:
            score = <float>results[j][1] + (<float>results[j][3] / 1000)
            if score > cmp_max_score:
                cmp_max_score = score
                i = j

    result_contents.insert(result_contents.end(),results[i][0].begin(),results[i][0].end())

    if i == -1:
        return 0


    return 1


FORMS = 'PRS'
DEF DEBUG = False
from time import sleep
cdef void tokenize_inner(
    char* word, 
    int cursor, 
    int length, 
    Parts parts, 
    vector[vector[Parts]]* cache,
    int max_call_depth, 
    float* max_score, 
    float min_score_ratio, 
    int call_depth,
    vector[Parts]* returns) nogil:

    cdef:
        vector[int] contents = parts[0]
        vector[int] new_contents
        vector[int]* temp_contents
        vector[int]* contents_tmp
        float score = parts[1]
        float new_score 
        int len_contents = contents.size()
        vector[Parts]* cache_value
        vector[Parts]* true_trie_value
        vector[Parts] to_be_cached, ret
        int cache_length
        Parts *temp_parts
        Parts  new_parts
        int A_part, B_part
        int temp_parts_length
        int i, j, k, m, n, p, _cursor, repeated, A_part_form, B_part_form
        vector[int] prefixes_ids
        int len_this, len_ret, len_p, len_p_part_max, len_part_max = parts[3]
        bint has_root
        
    if call_depth > max_call_depth:
        return

    # assert cache[0].size() > cursor, "cache[0].size() > cursor"
    cache_value = &cache[0][cursor]  
    cache_length = cache_value.size()

    if cache_length > 0:
        for i in range(cache_length):
            merge_two_parts(
                &parts,
                &(cache_value[0][i]),
                returns,
                &to_be_cached,
                call_depth,
                max_score,
                False,
                min_score_ratio
            )
    
        return

    cdef bint matched = False


    to_be_cached = vector[Parts]()
    to_be_cached.reserve(1024)

    len_this = length - cursor
    
    prefixes_ids = trie_obj._prefixes_id(
        word + cursor,        # pass the characters starting from `cursor` position
        len_this,      # pass the length after the `cursor` position
    )
    
    # iterate the prefixes
    # i.e. internal  ->  in, inter, intern
    for i from prefixes_ids.size()-1 >= i >= 0: 
        p = prefixes_ids[i]
        len_p = true_prefixes_lengths[p]
        
        matched = True
        true_trie_value = &true_trie_values[p]

        # iterate the possible parts
        # i.e. internal  ->  intern (R),           inter (P),   in (P), in (R), in (S)
        # i.e. reducing  ->  reduce (R) ing (S),   red (R),     re (P)
        for j in range(<int>true_trie_value.size()):
            temp_parts = &(true_trie_value[0][j])

            # split ends here
            # cursor + len_p == length
            if len_p == len_this or call_depth >= max_call_depth: 
                new_parts = (
                    temp_parts[0][0],
                    temp_parts[0][1],
                    cursor+len_p,
                    len_part_max if len_part_max > temp_parts[0][3] else temp_parts[0][3],
                )
                to_be_cached.push_back(new_parts)
                merge_two_parts(
                    &parts,
                    &new_parts,
                    returns,
                    &to_be_cached,
                    call_depth,
                    max_score,
                    True,
                    min_score_ratio
                )
                
                
            else:
                ret = vector[Parts]()
                ret.reserve(1024)

                tokenize_inner(
                    word, 
                    cursor+len_p, 
                    length, 
                    temp_parts[0], 
                    cache, 
                    max_call_depth, 
                    max_score,
                    min_score_ratio,
                    call_depth+1, 
                    &ret
                )
                len_ret = ret.size()
                if len_ret > 0:
                    for k in range(len_ret):
                        merge_two_parts(
                            &parts,
                            &ret[k],
                            returns,
                            &to_be_cached,
                            call_depth,
                            max_score,
                            True,
                            min_score_ratio
                        )
                        

                        
    repeated = 0
    while True:
        _cursor = cursor+repeated
        if _cursor > 0 and _cursor < length - 1 and word[_cursor] == word[_cursor-1]:
            repeated += 1
            continue
        break
    if repeated > 0:
        pass
    else:
        new_parts = (vector[int](), 0, 0, 0)
        if true_all_alphabets.find(word[cursor]) != true_all_alphabets.end():
            new_parts[0].push_back(
                get_part(true_all_alphabets[word[cursor]], ROOT, MERGE_MODE_BOTH)
            )
    if not matched:
        ret = vector[Parts]()
        ret.reserve(1024)
        tokenize_inner(
            word, 
            cursor+repeated, 
            length, 
            new_parts, 
            cache, 
            max_call_depth, 
            max_score,
            min_score_ratio,
            call_depth+1, 
            &ret
        )

        # below are copied from above
        
        len_ret = ret.size()
        
        if len_ret > 0:
            for k in range(len_ret):
                temp_parts = &ret[k]

                temp_contents = &(temp_parts[0][0])

                merge_two_parts(
                    &parts,
                    &ret[k],
                    returns,
                    &to_be_cached,
                    call_depth,
                    max_score,
                    True,
                    min_score_ratio
                )



    cache[0][cursor] = to_be_cached
    


cdef void merge_two_parts(
    Parts* A_parts, 
    Parts* B_parts, 
    vector[Parts]* returns, 
    vector[Parts]* to_be_cached, 
    int call_depth,
    float* max_score, 
    bint do_cache,
    float min_score_ratio
    ) nogil:
    cdef:
        vector[int]* A_contents = &(A_parts[0][0])
        vector[int]* B_contents = &(B_parts[0][0])
        vector[int]* contents_tmp

        float A_score = A_parts[0][1]
        float B_score = B_parts[0][1]
        float new_score

        int A_length = A_contents[0].size()
        int B_length = B_contents[0].size()

        int A_part, B_part

        int A_part_form

        int len_A_part_max = A_parts[0][3]
        int len_B_part_max = B_parts[0][3]

        int m

        bint has_root

        vector[int] new_contents


    if A_length > 0:
        A_part = A_contents[0][A_length-1]
        B_part = B_contents[0][0]
        if adjacent_violate_c(A_part, B_part):
            return

    if do_cache:
        to_be_cached[0].push_back(B_parts[0])


    new_contents = vector[int]()
    new_score = A_score + B_score
    if A_length > 0:
        new_score -= 1
    has_root = False

    if call_depth != 0:
        new_contents.reserve(A_length+B_length)
        for m in range(A_length):
            new_contents.push_back(A_contents[0][m])
        for m in range(B_length):
            new_contents.push_back(B_contents[0][m])
        
    else:
        # The final split here, 
        new_contents.reserve(A_length+B_length+4)

        for m in range(A_length):
            if part_id_to_irregular_parts.find(A_contents[0][m]) == part_id_to_irregular_parts.end():
                new_contents.push_back(A_contents[0][m])
            else:
                contents_tmp = &(part_id_to_irregular_parts[A_contents[0][m]][0])
                for m in range(contents_tmp.size()):
                    new_contents.push_back(contents_tmp[0][m])

        for m in range(B_length):
            if part_id_to_irregular_parts.find(B_contents[0][m]) == part_id_to_irregular_parts.end():
                new_contents.push_back(B_contents[0][m])
            else:
                contents_tmp = &(part_id_to_irregular_parts[B_contents[0][m]][0])
                for m in range(contents_tmp.size()):
                    new_contents.push_back(contents_tmp[0][m])

        # Prefer splits that have a root
        # if no root, punish! 
        for m in range(A_length+B_length):
            A_part = new_contents[m]
            A_part_form = A_part % 3
            if A_part_form == ROOT:
                has_root = True
                break
        if not has_root:
            new_score -= 1

        # ends with prefix and start with suffix is not good
        for m from A_length+B_length-1 >= m >= 0: 
            A_part = new_contents[m]
            A_part_form = A_part % 3
            if A_part_form == PREFIX:
                new_score -= 1
            elif A_part_form == SUFFIX:
                continue
            break
        for m in range(A_length+B_length):
            A_part = new_contents[m]
            A_part_form = A_part % 3
            if A_part_form == SUFFIX:
                new_score -= 1
            elif A_part_form == PREFIX:
                continue
            break

        if new_score > max_score[0]:
            max_score[0] = new_score
    if new_score < max_score[0] * min_score_ratio:
        return


    returns.push_back(
        (new_contents, 
        new_score, 
        B_parts[0][2], 
        len_A_part_max if len_A_part_max > len_B_part_max else len_B_part_max)
        ) 