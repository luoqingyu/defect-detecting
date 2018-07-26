from multiprocessing import Pool
from multiprocessing  import Manager
import os, time, random

def long_time_task(name,dict):
    dict[name] =name


if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    mydict = Manager().dict()  # 主进程与子进程共享这个字典
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,mydict))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
    print(mydict)
