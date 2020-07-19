# cython: language_level=2

cdef extern from "lrucache.hpp":
    cdef cppclass lru_cache[T, U] nogil:
        lru_cache(size_t) nogil
        void put(const T&, const U&) nogil
        const U& get(const T&) nogil
        bint exists(const T& ) nogil
        size_t size() nogil

