import os
import argparse

parser = argparse.ArgumentParser(
                    prog="reproduce.py",
                    description="Usage of the automated setup",
                    epilog="Text at the bottom of help")

parser.add_argument("-a", "--reproduce-all", action="store_true", help="reproduce the whole project (matrices, results, paper)")
parser.add_argument("-r", "--reproduce-results", action="store_true", help="reproduce all results (plots/tables)")
parser.add_argument("-p", "--reproduce-paper", action="store_true", help="compile the paper (LaTeX)")

args = parser.parse_args()

if args.reproduce_results or args.reproduce_all:
    os.system("python setup/reproduce_results.py")
if args.reproduce_paper or args.reproduce_all:
    os.system("python setup/reproduce_paper.py")