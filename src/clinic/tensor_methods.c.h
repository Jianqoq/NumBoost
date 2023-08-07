/*[clinic input]
preserve
[clinic start generated code]*/

#if defined(Py_BUILD_CORE) && !defined(Py_BUILD_CORE_MODULE)
#  include "pycore_gc.h"            // PyGC_Head
#  include "pycore_runtime.h"       // _Py_ID()
#endif


PyDoc_STRVAR(backward__doc__,
"backward($self, grad, /)\n"
"--\n"
"\n"
"args can be nd.array or Tensor obj.\n"
"\n"
"  args\n"
"    The object to be pickled.");

#define BACKWARD_METHODDEF    \
    {"backward", (PyCFunction)backward, METH_O, backward__doc__},
/*[clinic end generated code: output=3de2325d22e28ad9 input=a9049054013a1b77]*/
