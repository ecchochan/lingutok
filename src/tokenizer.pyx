# cython: profile=False, embedsignature=True, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
#distutils: language = c++

from __future__ import unicode_literals

from std_iostream cimport stringstream, istream, ostream
from libc.string cimport strncmp
cimport keyset
cimport key
cimport agent
cimport trie
cimport iostream
cimport base

import itertools
import struct
import warnings

from libcpp.vector cimport vector

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

    cdef int _get_id(self, agent.Agent& ag):
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


cdef class Trie(_UnicodeKeyedTrie):
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

    cdef vector[int] _prefixes_id(self, bytes key):
        """
        Return a list with all prefixes of a given key.
        """
        # this an inlined version of ``list(self.iter_prefixes(key))``

        cdef vector[int] res
        cdef agent.Agent ag
        ag.set_query(key, len(key))

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




















from pkg_resources import resource_filename

from libcpp.vector cimport vector

package_name = 'LinguisticTokenizer'
import os.path

def get_file(x):
    return resource_filename(package_name, 'resources/' + x)

    
if not os.path.isfile(get_file('words2.txt')):
    raise Exception('Resource files not found.')


fn = get_file('words2.txt')
with open(fn) as f:
    words = f.read()
    

    
import json


single_words_only = set()
single_words_any  = set()
with open(get_file('single_words.txt')) as f:
    for line in f:
        line = line.strip()
        if line.startswith('#'):
            continue
        if line.startswith('*'):
            single_words_any.add(line)
            continue
            
        if not line:
            continue

        single_words_only.add(line)

with open(get_file('irreguler_exceptions.json')) as f:
    irreguler_exceptions = json.load(f)

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
            if e in irreguler_exceptions:
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
            E = e.strip('-')
            if E not in all_parts:
                all_parts[E] = part_id
                all_parts_list.append(E)
                part_id += 1
            
            
            
            assert '+' not in e
            index += 1
            if not e:
                continue
                
            if token_id > 0:
                all_vocabs[e] = token_id - 1
                all_vocabs_list.append(e)
                
                
            if e.startswith('-'):
                e = e[1:]
                if e in single_words_only:
                    single_words_any.add(e)
                    single_words_only.remove(e)
                if e not in suffixes:
                    suffixes[e] = set()
                    
                    if not c:
                        suffixes[e].add('')
                if b and index == 1:
                    for bb in b.split(','):
                        bb = bb.strip()
                        assert bb.startswith('+')
                        suffixes[e].add(bb[1:])
            elif e.endswith('-'):
                e = e[:-1]
                if e in single_words_only:
                    single_words_any.add(e)
                    single_words_only.remove(e)
                    
                prefixes.add(e)
            else:
                roots.add(e)
for e in single_words_only:
    assert '-' not in e
    assert e
    if e not in all_parts:
        all_parts[e] = part_id
        all_parts_list.append(e)
        part_id += 1
        if e not in roots:
            roots.add(e)
        
    
import rules
    
# A no prefixed by B
# all roots
no_prefixed_by_of = rules.expand_allowed_reverse
    
# A only suffixed by B
only_suffixed_by_of = {k:v or set() for k, v in rules.expand_allowed.items()}
rules.expand_allowed.clear()

# A no suffixed by B
no_suffixed_by_of = rules.pairs_exceptions

# A no suffixed by B (root)
no_suffixed_by_of_roots = {
    k[0]: set(v.split(','))
    for k, v in rules.expand_exceptions.items() if k[1]
}

# A no suffixed by B (non-root)
no_suffixed_by_of_nonroots = {
    k[0]: set(v.split(','))
    for k, v in rules.expand_exceptions.items() if not k[1]
}
        
rules.expand_exceptions.clear()

# len(prefixes), len(suffixes), len(roots), len(all_parts), len(all_vocabs)




def adjacent_violate(A, B, A_is_root):

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



def get_morphed_suffixes_merge_suffixes(int min_left = 1):
    cdef int len_s, len_p
    cdef float score
    for suffix, prefixers in suffixes.items():
        for p in prefixers:
            if not p:
                continue
            for s in suffixes:
                if s not in all_parts or s in irreguler_exceptions:
                    continue
                if s.endswith(p):
                    A = s
                    B = suffix
                        
                    if (A in single_words_any or A in single_words_only) != (B in single_words_any or B in single_words_only):
                        continue
                    if adjacent_violate(A, B, False):
                        continue
                        
                    len_s = len(s)
                    len_p = len(p)
                        
                    if len_s - len_p < min_left:
                        continue
                        
                    
                    overlapped = s[len_s-len_p:]
                    
                    if overlapped == 'e' or overlapped == 'o' or overlapped == 'y':
                        score = 0
                    else:
                        score = -len_p
                    
                    
                    yield (A,B, s[:len_s-len_p]+suffix, score )
                
                

cdef struct AB:
    int A
    int B
    
def get_morphed_roots_merge_suffixes(int min_left = 1):
    cdef int len_s, len_p
    cdef float score
    for suffix, prefixers in suffixes.items():
        for p in prefixers:
            if not p:
                continue
            for s in roots:
                if s not in all_parts or s in irreguler_exceptions:
                    continue
                if s.endswith(p):
                    A = s
                    B = suffix
                        
                    if (A in single_words_any or A in single_words_only) != (B in single_words_any or B in single_words_only):
                        continue
                    if adjacent_violate(A, B, True):
                        continue
                        
                    len_s = len(s)
                    len_p = len(p)
                        
                    if len_s - len_p < min_left:
                        continue
                        
                        
                    overlapped = s[len_s-len_p:]
                    
                    if overlapped == 'e' or overlapped == 'o' or overlapped == 'y':
                        score = 0
                    else:
                        score = -len_p
                        
                    yield (A,B, s[:-len(p)]+suffix, score )

def mix_two_morphed(h, k, min_left):
    bucket = []
    temp_index = {}
    for e in k:
        A, B, morphed, score = e
        if len(morphed) - len(B) < min_left:
            continue
        if A not in temp_index:
            temp_index[A] = []
            
        temp_index[A].append(e)

    for A, B, morphed, score in h:
        if B in temp_index:
            for C, D, morphed2, score2 in temp_index[B]:
                new_morphed = morphed[:-len(B)]+ morphed2
                yield (A, B, D, new_morphed, score+score2)

                
                
                

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
                
                
                
                












from libcpp.unordered_map cimport unordered_map


ctypedef enum PartType:
    PREFIX,
    ROOT,
    SUFFIX
    
cdef struct Part:
    int contents
    float score
    bint individual
    
    
cdef struct Parts:
    vector[Part] contents
    float score
    

cdef float ROOT_PUNISHMENT = 0.5



trie_values = {}

seen = {}

def gen():
    def get_bucket(e):
        if e not in trie_values:
            bucket = []
            trie_values[e] = bucket
            return bucket
        return trie_values[e] 
    
    def get_part(part, mode, individual):
        key = (part, mode, individual)
        if key in seen:
            return seen[key]
        
        ret = seen[key] = Part(part, mode, individual)
        return ret
    
    for e in prefixes:
        parts = Parts(
            [get_part(all_parts[e], PartType.PREFIX, e in single_words_any or e in single_words_only)], 
            0
        )
        get_bucket(e).append(parts)
        
        assert e
        yield e
    for e in roots:
        parts = Parts(
            [get_part(all_parts[e], PartType.ROOT, e in single_words_any or e in single_words_only)], 
            -ROOT_PUNISHMENT
        )
        get_bucket(e).append(parts)
        assert e
        yield e
        
    for e in suffixes:
        parts = Parts(
            [get_part(all_parts[e], PartType.SUFFIX, e in single_words_any or e in single_words_only)], 
            0
        )
        get_bucket(e).append(parts)
        assert e
        yield e
        
        
    morphed_suffixes_merge_suffixes = list(get_morphed_suffixes_merge_suffixes())
    morphed_roots_merge_suffixes = list(get_morphed_roots_merge_suffixes())
    
    
    for A, B, merged, punishment in morphed_suffixes_merge_suffixes:
        parts = Parts(
            [
                get_part(all_parts[A], PartType.SUFFIX, False),
                get_part(all_parts[B], PartType.SUFFIX, False)
            ], 
            0
        )
        get_bucket(merged).append(parts)
        assert merged
        yield merged
        
    for A, B, merged, punishment in morphed_roots_merge_suffixes:
        if (A in single_words_any or A in single_words_only) != (B in single_words_any or B in single_words_only):
            continue
        if adjacent_violate(A, B, True):
            continue
        parts = Parts(
            [
                get_part(all_parts[A], PartType.ROOT, False),
                get_part(all_parts[B], PartType.SUFFIX, False)
            ], 
            -ROOT_PUNISHMENT
        )
        get_bucket(merged).append(parts)
        assert merged
        yield merged
        
        
    for A, B, C, merged, punishment in mix_two_morphed(morphed_suffixes_merge_suffixes, morphed_suffixes_merge_suffixes, 1):
        parts = Parts(
            [
                get_part(all_parts[A], PartType.SUFFIX, False),
                get_part(all_parts[B], PartType.SUFFIX, False),
                get_part(all_parts[C], PartType.SUFFIX, False),
            ], 
            0
        )
        get_bucket(merged).append(parts)
        assert merged
        yield merged
    for A, B, C, merged, punishment in mix_two_morphed(morphed_roots_merge_suffixes, morphed_suffixes_merge_suffixes, 2):
        if (A in single_words_any or A in single_words_only) != (B in single_words_any or B in single_words_only):
            continue
        if (B in single_words_any or A in single_words_only) != (C in single_words_any or B in single_words_only):
            continue
        if adjacent_violate(A, B, True):
            continue
        if adjacent_violate(B, C, False):
            continue
        parts = Parts(
            [
                get_part(all_parts[A], PartType.ROOT, False),
                get_part(all_parts[B], PartType.SUFFIX, False),
                get_part(all_parts[C], PartType.SUFFIX, False),
            ], 
            -ROOT_PUNISHMENT
        )
        get_bucket(merged).append(parts)
        assert merged
        yield merged
