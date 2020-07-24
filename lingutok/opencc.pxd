# cython: language_level=2

cdef extern from "<opencc.h>" nogil:
    ctypedef void* opencc_t
    opencc_t opencc_open(const char* configFileName) nogil
    char* opencc_convert_utf8(opencc_t opencc, const char* input, size_t length) nogil
    void opencc_convert_utf8_free(char* str) nogil