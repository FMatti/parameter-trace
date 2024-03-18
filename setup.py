import os
import argparse

parser = argparse.ArgumentParser(
                    prog="setup.py",
                    description="Usage of the automated setup",
                    epilog="Text at the bottom of help")

parser.add_argument("-a", "--generate-all", action="store_true", help="reproduce the whole project (matrices, ...)")
parser.add_argument("-m", "--generate-matrices", action="store_true", help="download and create all matrices")

args = parser.parse_args()

if args.generate_matrices or args.generate_all:
    os.system("python setup/generate_matrices.py")
