# cython: language_level=2

cdef extern from "<utf8proc.h>" nogil:

    ctypedef enum utf8proc_option_t:
        UTF8PROC_NULLTERM  = (1<<0),
        UTF8PROC_STABLE    = (1<<1),
        UTF8PROC_COMPAT    = (1<<2),
        UTF8PROC_COMPOSE   = (1<<3),
        UTF8PROC_DECOMPOSE = (1<<4),
        UTF8PROC_IGNORE    = (1<<5),
        UTF8PROC_REJECTNA  = (1<<6),
        UTF8PROC_NLF2LS    = (1<<7),
        UTF8PROC_NLF2PS    = (1<<8),
        UTF8PROC_STRIPCC   = (1<<9),
        UTF8PROC_CASEFOLD  = (1<<10),
        UTF8PROC_CHARBOUND = (1<<11),
        UTF8PROC_LUMP      = (1<<12),
        UTF8PROC_STRIPMARK = (1<<13),
        UTF8PROC_STRIPNA    = (1<<14),
    
    ssize_t utf8proc_map(
        const unsigned char *str, ssize_t strlen, unsigned char **dstptr, utf8proc_option_t options
    ) nogil
    unsigned char *utf8proc_NFD(const unsigned char *str) nogil
    unsigned char *utf8proc_NFC(const unsigned char *str) nogil
    unsigned char *utf8proc_NFKD(const unsigned char *str) nogil
    unsigned char *utf8proc_NFKC(const unsigned char *str) nogil

cdef inline unsigned char *utf8proc_NFKD_strip(const unsigned char *str):
  cdef unsigned char *retval;
  utf8proc_map(str, 0, &retval, <utf8proc_option_t>(UTF8PROC_NULLTERM | UTF8PROC_STABLE |
    UTF8PROC_DECOMPOSE | UTF8PROC_COMPAT | UTF8PROC_STRIPMARK | UTF8PROC_STRIPNA | UTF8PROC_IGNORE));
  return retval;


cdef inline unsigned char *utf8proc_NFKC_strip(const unsigned char *str):
  cdef unsigned char *retval;
  utf8proc_map(str, 0, &retval, <utf8proc_option_t>(UTF8PROC_NULLTERM | UTF8PROC_STABLE |
    UTF8PROC_COMPOSE | UTF8PROC_COMPAT | UTF8PROC_STRIPMARK | UTF8PROC_STRIPNA | UTF8PROC_IGNORE));
  return retval;

