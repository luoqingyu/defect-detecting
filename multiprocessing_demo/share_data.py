import multiprocessing
from multiprocessing import Pool
import os, time, random

def func(mydict, mylist):
    mydict["index1"] = "aaaaaa"  # 子进程改变dict,主进程跟着改变
    mydict["index2"] = "bbbbbb"
    mylist.append(11)  # 子进程改变List,主进程跟着改变
    mylist.append(22)
    mylist.append(33)
    mylist[1]=18
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    x = range(80)
    mydict = multiprocessing.Manager().dict()  # 主进程与子进程共享这个字典
    mylist = multiprocessing.Manager().list(x)  # 主进程与子进程共享这个List



    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')

    print(mylist)
    print(mydict)
