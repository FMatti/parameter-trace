import __context__

import os
import glob

plot_files = glob.glob("paper/plots/*.py")
table_files = glob.glob("paper/tables/*.py")

for plot_file in plot_files:
    os.system("python " + plot_file)

for table_file in table_files:
    os.system("python " + table_file)
