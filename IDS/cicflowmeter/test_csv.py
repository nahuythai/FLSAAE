import time
from optparse import OptionParser
SLEEP_INTERVAL = 1.0


def readlines_then_tail(fin):
    "Iterate through lines and then tail for further lines."
    while True:
        line = fin.readline()
        if line:
            yield line
        else:
            tail(fin)


def tail(fin):
    "Listen for new lines added to file."
    while True:
        where = fin.tell()
        line = fin.readline()
        if not line:
            time.sleep(SLEEP_INTERVAL)
            fin.seek(where)
        else:
            yield line


def main():
    p = OptionParser("usage: tail.py file")
    (options, args) = p.parse_args()
    if len(args) < 1:
        p.error("must specify a file to watch")
    with open(args[0], 'r') as fin:
        for line in readlines_then_tail(fin):
            print(line.strip())


if __name__ == '__main__':
    main()