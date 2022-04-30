import re
import sys

def pckt_dict(path, parti, fnd):
    '''Extract pckts per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    patt = re.compile("I00(.*)[^I00]*"+fnd)
    d = dict()
    for i in patt.findall(text):
        d[int(i[:6])//parti] = d.setdefault(int(i[:6])//parti, 0) + 1
    return(d)

def byte_dict(path, parti, fnd):
    '''Extract bytes per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    patt = re.compile("I00(.*)[^I00]*"+fnd+"(.*)\n")
    d = dict()
    for i in patt.findall(text):
        d[int(i[0][:6])//parti] = d.setdefault(int(i[0][:6])//parti, 0) + int(re.search(" (\d*) bytes", i[1]).group(1))
    return(d)

if __name__ == '__main__':
    if (len(sys.argv) not in [5]):
        sys.exit("speedlog.py:\nUsage: python3 speedlog.py [file with stderr of CLI/SRV: PATH] [mode: BYTE|PCKT] [direction: SENT|RCVD] [time partition in milliseconds: INT_NUMBER]")
    else:
        filepath = sys.argv[1]
        fnd = "Sent packet:" if sys.argv[3].upper() == "SENT" else "Received packet:"
        try:
            parti = int(sys.argv[4])
        except Exception as E:
            sys.exit(f"speedlog.py:\nPartition must be integer: {E}")
        print(f"Estimating speed in {sys.argv[2].upper()} mode:")
        if sys.argv[2].upper() == "BYTE":
            d = byte_dict(filepath, parti, fnd)
            for i, j in d.items():
                print(f"Second {i*parti/1000}-{(i+1)*parti/1000}: Bytes {fnd.split()[0].lower()}: {j}")
            if d:
                print(f"Mean speed {sum([j for i, j in d.items()])/len([j for i, j in d.items()])} per {parti/1000} seconds")
        elif sys.argv[2].upper() == "PCKT":
            d = pckt_dict(filepath, parti, fnd)
            for i, j in d.items():
                print(f"Second {i*parti/1000}-{(i+1)*parti/1000}: Packets {fnd.split()[0].lower()}: {j}")
        else:
            sys.exit("speedlog.py:\nNo such mode")
        sys.exit("Success")