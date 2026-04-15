#ifndef _CONFIGURATION_HEADER_GUARD_H_
#define _CONFIGURATION_HEADER_GUARD_H_

/* zlib/zlib-ng disabled: zmat uses miniz instead */
/* #undef HAVE_ZLIB */
/* #undef HAVE_ZLIB_NG */
#define HAVE_ZSTD TRUE
/* #undef HAVE_IPP */
/* #undef BLOSC_DLL_EXPORT */
#define HAVE_PLUGINS TRUE

#endif
