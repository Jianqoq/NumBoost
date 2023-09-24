#ifndef OP_H
#define OP_H
typedef enum {
  ADD,
  SUB,
  MUL,
  DIV,
  MOD,
  POW,
  LSHIFT,
  RSHIFT,
  FLOOR_DIV,
  BITWISE_AND,
  BITWISE_OR,
  BITWISE_XOR,
  BITWISE_INVERT,
  SQUARE,
  DIVMOD,
  LT,
  GT,
  EQ,
  NEQ,
  WHERE,
  /*====== seperator for binary and elementwise operation ======*/
  POSITIVE,
  NEGATIVE,
  SIN,
  COS,
  TAN,
  ARCSIN,
  ARCCOS,
  ARCTAN,
  SINH,
  COSH,
  TANH,
  ARCSINH,
  ARCCOSH,
  ARCTANH,
  SQRT,
  EXP,
  LOG,
  LOG10,
  ABS
} op_type;
typedef enum {
  SUM,
  PROD,
  MAX,
  MIN,
  ANY,
  ALL,
  MEAN,
  ARGMIN,
  ARGMAX
} reduction_op_type;
// typedef enum {
//   NB_BOOL,
//   NB_BYTE,
//   NB_UBYTE,
//   NB_SHORT,
//   NB_USHORT,
//   NB_INT,
//   NB_UINT,
//   NB_LONG,
//   NB_ULONG,
//   NB_LONGLONG,
//   NB_ULONGLONG,
//   NB_FLOAT,
//   NB_DOUBLE,
//   NB_LONGDOUBLE,
//   NB_HALF
// } nb_dtype;

// typedef signed char nb_byte;
// typedef unsigned char nb_ubyte;
// typedef unsigned short nb_ushort;
// typedef unsigned int nb_uint;
// typedef unsigned long nb_ulong;

// typedef char nb_char;
// typedef short nb_short;
// typedef int nb_int;
// typedef long nb_long;
// typedef float nb_float;
// typedef double nb_double;

#endif