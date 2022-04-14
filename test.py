
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default = None)
args = parser.parse_args()

if args.mode == "1":
    parser.add_argument("--para", nargs='+', default = None)  
    print(args.para)
elif args.mode == "2":
    parser.add_argument("--name", default = "hello")
    print(args.name)
else:
    parser.add_argument("--cost", default = 0)
    
print(args.para)