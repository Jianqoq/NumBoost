from numpy import dtype as _dtype
import platform
if platform.system() == 'Windows':
    int32 = 7
    int64 = 9
    uint32 = 8
    uint64 = 10
else:
    int32 = 5
    int64 = 7
    uint32 = 6
    uint64 = 8

bool_ = 0
int8 = 1
int16 = 3
uint8 = 2
uint16 = 4
float16 = 23
float32 = 11
float64 = 12
longdouble = 13
double = 12
float_ = 12
longlong = 9
ulonglong = 10
short = 3
ushort = 4
byte = 1
ubyte = 2
int_ = 7
uint = 8
