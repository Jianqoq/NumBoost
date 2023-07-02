#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"

// Function to create a stack of given capacity.
Stack *createStack(uint64_t capacity)
{
    Stack *stack = (Stack *)malloc(sizeof(Stack));
    stack->len = 0;
    stack->index = -1;
    stack->array = (Tuple *)malloc(capacity * sizeof(Tuple));
    stack->max_len = capacity;
    return stack;
}

// Stack is full when top is equal to the last index
int isFull(Stack *stack)
{
    return stack->len == stack->max_len;
}

// Stack is empty when top is equal to -1
int isEmpty(Stack *stack)
{
    return stack->len == 0;
}

// Function to add an item to stack.  It increases top by 1
PyObject *push(Stack *stack, Tuple item)
{
    if (isFull(stack))
    {
        PyErr_SetString(PyExc_RuntimeError, "Stack overflowed");
                PyErr_Print();
        PyErr_Clear();
        Py_Finalize();
        printf("Stack overflowed\n");
        return NULL;
    }
    stack->len++;
    stack->array[++stack->index] = item;
    return Py_None;
}

// Function to remove an item from stack.  It decreases top by 1
Tuple pop(Stack *stack)
{
    if (isEmpty(stack))
    {
        PyErr_SetString(PyExc_ValueError, "Negative value not allowed");
        PyErr_Print();
        PyErr_Clear();
    }
    stack->len--;
    return stack->array[stack->index--];
}
