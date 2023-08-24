from numpy import dtype as _dtype
import platform
if platform.system() == 'Windows':
    uint = 6
    uint32 = 7
else:
    uint = 10
    uint32 = 11

if _dtype('longdouble').itemsize == 16:
    longdouble = 13
else:
    longdouble = 12

bool_ = 0
byte = 1
ubyte = 2
short, int16 = 3, 3
ushort = 4
int_, int32 = 5, 5
long, int64 = 7, 7
ulong, uint64 = 8, 8
longlong = 9
ulonglong = 10
float32 = 11
float = 11
float64 = 12
double = 12
float16 = 23
