import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{color}[{name}] done in {et:.0f} s{nocolor}".format(
        name=name, et=time.time() - t0,
        color='\033[1;33m', nocolor='\033[0m'))
