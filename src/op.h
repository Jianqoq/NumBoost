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
  POSITIVE,
  NEGATIVE,
  SQUARE,
  /*====== seperator for binary and elementwise operation ======*/
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
  NB_BOOL,
  NB_BYTE,
  NB_UBYTE,
  NB_SHORT,
  NB_USHORT,
  NB_INT,
  NB_UINT,
  NB_LONG,
  NB_ULONG,
  NB_LONGLONG,
  NB_ULONGLONG,
  NB_FLOAT,
  NB_DOUBLE,
  NB_LONGDOUBLE,
  NB_HALF
} nb_dtype;

#endif