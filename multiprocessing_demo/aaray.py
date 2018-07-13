import multiprocessing
import numpy as np

def func(num):
    num[2] = 9999  # 子进程改变数组，主进程跟着改变


if __name__ == "__main__":
    x =  range(80)
    print(type(x))
    num = multiprocessing.Array("i", x)  # 主进程与子进程共享这个数组
    num2 = multiprocessing.Array("i", x)  # 主进程与子进程共享这个数组
    print(num[:])
    p = multiprocessing.Process(target=func, args=(num,))
    p.start()
    p.join()


    print(num.append(num2))