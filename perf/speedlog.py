import re
import sys
from collections import defaultdict
import argparse


def pckt_dict(path, parti, fnd):
    '''Extract pckts per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd)
    d = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[:6]) // parti
        d[time_period] += 1
    return(d)


def byte_dict(path, parti, fnd):
    '''Extract bytes per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd+"(.*)\n")
    d = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[0][:6]) // parti
        d[time_period] += int(re.search(r" (\d*) bytes", i[1]).group(1))
    return(d)


arg_parser = argparse.ArgumentParser(prog='speedlog',
                                     description='Calculate amount of client data sent. ' 
                                     'Note, that total packets calculated can be more, than content delivered, '
                                     'because we calculate content size of packets, that include content. ')                                     
arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.1')

arg_parser.add_argument('path', metavar='PATH', nargs='?', type=str, default='',
                        help='Path of file with stderr of ngtcp2 client|server.')
arg_parser.add_argument('mode', metavar='MODE', nargs='?', type=str, default='',
                        choices=['BYTE', 'PCKT'], help='Specify mode of calculation. '
                        'Note, that these are UDP packets! ')
arg_parser.add_argument('direction', metavar='DIRECTION', nargs='?', type=str, default='',
                        choices=['SENT', 'RCVD'], help='Direction of packets calculated.')
arg_parser.add_argument('parti', metavar='PARTITION', nargs='?', type=str, default='',
                        help='Integer number, that provides time partition.')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    filepath = args.path
    fnd = "Sent packet:" if args.direction.upper() == "SENT" else "Received packet:"
    try:
        parti = int(args.parti)
    except Exception as E:
        sys.exit(f"speedlog.py:\nPartition must be integer: {E}")
    print(f"Estimating speed in {args.mode.upper()} mode:")
    if args.mode.upper() == "BYTE":
        d = byte_dict(filepath, parti, fnd)
        for i, j in d.items():
            print(f"Second {i*parti/1000}-{(i+1)*parti/1000}: Bytes {fnd.split()[0].lower()}: {j}")
        if d:
            print(f"Mean speed {sum([j for i, j in d.items()][:-1])/len([j for i, j in d.items()][:-1])} bytes per {parti/1000} seconds")
    elif args.mode.upper() == "PCKT":
        d = pckt_dict(filepath, parti, fnd)
        for i, j in d.items():
            print(f"Second {i*parti/1000}-{(i+1)*parti/1000}: Packets {fnd.split()[0].lower()}: {j}")
        if d:
            print(f"Mean speed {sum([j for i, j in d.items()][:-1])/len([j for i, j in d.items()][:-1])} pckts per {parti/1000} seconds")
    else:
        sys.exit("speedlog.py:\nNo such mode")
    sys.exit("Success")