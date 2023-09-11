#ifndef NUMBOOST_MATH_H
#define NUMBOOST_MATH_H
#include <numpy/npy_math.h>

#define nb_sin npy_sin
#define nb_sini
#define nb_sinf npy_sinf
#define nb_sinl npy_sinl

#define nb_cos npy_cos
#define nb_cosi
#define nb_cosf npy_cosf
#define nb_cosl npy_cosl

#define nb_tan npy_tan
#define nb_tani
#define nb_tanf npy_tanf
#define nb_tanl npy_tanl

#define nb_arcsin npy_arcsin
#define nb_arcsini
#define nb_arcsinf npy_arcsinf
#define nb_arcsinl npy_arcsinl

#define nb_arccos npy_arccos
#define nb_arccosi
#define nb_arccosf npy_arccosf
#define nb_arccosl npy_arccosl

#define nb_arctan npy_arctan
#define nb_arctani
#define nb_arctanf npy_arctanf
#define nb_arctanl npy_arctanl

#define nb_arctan2 npy_arctan2
#define nb_arctan2i
#define nb_arctan2f npy_arctan2f
#define nb_arctan2l npy_arctan2l

#define nb_sinh npy_sinh
#define nb_sinhi
#define nb_sinhf npy_sinhf
#define nb_sinhl npy_sinhl

#define nb_cosh npy_cosh
#define nb_coshi
#define nb_coshf npy_coshf
#define nb_coshl npy_coshl

#define nb_tanh npy_tanh
#define nb_tanhi
#define nb_tanhf npy_tanhf
#define nb_tanhl npy_tanhl

#define nb_arcsinh npy_arcsinh
#define nb_arcsinhi
#define nb_arcsinhf npy_arcsinhf
#define nb_arcsinhl npy_arcsinhl

#define nb_arccosh npy_arccosh
#define nb_arccoshi
#define nb_arccoshf npy_arccoshf
#define nb_arccoshl npy_arccoshl

#define nb_arctanh npy_arctanh
#define nb_arctanhi
#define nb_arctanhf npy_arctanhf
#define nb_arctanhl npy_arctanhl

#define nb_modi(x, y) ((y) == 0 ? 0 : (x) % (y))
#define nb_mod(x, y) ((y) == 0 ? 0 : npy_fmod((x), (y)))
#define nb_modf(x, y) ((y) == 0 ? 0 : npy_fmodf((x), (y)))
#define nb_modl(x, y) ((y) == 0 ? 0 : npy_fmodl((x), (y)))

#define nb_floori(x) (x)
#define nb_floor(x) npy_floor((x))
#define nb_floorf(x) npy_floorf((x))
#define nb_floorl(x) npy_floorl((x))

#define nb_add(x, y) ((x) + (y))
#define nb_addi(x, y) ((x) + (y))
#define nb_addf(x, y) ((x) + (y))
#define nb_addl(x, y) ((x) + (y))

#define nb_sub(x, y) ((x) - (y))
#define nb_subi(x, y) ((x) - (y))
#define nb_subf(x, y) ((x) - (y))
#define nb_subl(x, y) ((x) - (y))

#define nb_mul(x, y) ((x) * (y))
#define nb_muli(x, y) ((x) * (y))
#define nb_mulf(x, y) ((x) * (y))
#define nb_mull(x, y) ((x) * (y))

#define nb_div(x, y) ((x) / (y))
#define nb_divi(x, y) ((x) / (y))
#define nb_divf(x, y) ((x) / (y))
#define nb_divl(x, y) ((x) / (y))

#define nb_fdivi(x, y) ((y) == 0 ? 0 : ((x) / (y)))
#define nb_fdivf(x, y) ((y) == 0 ? NAN : npy_floorf(((x) / (y))))
#define nb_fdiv(x, y) ((y) == 0 ? NAN : npy_floor((x) / (y)))
#define nb_fdivl(x, y) ((y) == NAN ? 0 : npy_floorl(((x) / (y))))

#endif // NUMBOOST_MATH_H
