import __context__

import os
import glob

plot_files = glob.glob("paper/plots/*.py")
table_files = glob.glob("paper/tables/*.py")

for plot_file in plot_files:
    os.system("python " + plot_file)

for table_file in table_files:
    os.system("python " + table_file)

project_name = "paper"
os.chdir(os.path.join(os.path.abspath(""), project_name))

def send_command(command, file="", args=""):
    cmd = " ".join([command, file, *args])
    os.system(cmd)

pdflatex_args = ["-synctex=1", "-interaction=nonstopmode", "-file-line-error"]

send_command("pdflatex", project_name, pdflatex_args)
send_command("bibtex", project_name)
send_command("makeglossaries", project_name)
send_command("pdflatex", project_name, pdflatex_args)
send_command("pdflatex", project_name, pdflatex_args)