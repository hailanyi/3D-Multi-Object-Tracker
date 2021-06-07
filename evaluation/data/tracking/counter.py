import os
import re

def count(name):
    sum=0
    with open(name) as f:

        for line in f.readlines():
            value=re.split(" ",line)
            print(len(value))
            sum+=float(value[-1])
    print(sum)

if __name__ == '__main__':
    name="evaluate_tracking.seqmap.test"
    count(name)