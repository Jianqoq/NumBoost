#ifndef OP_H
#define OP_H
typedef enum
{
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
} op_type;
#endif