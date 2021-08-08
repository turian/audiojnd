#!/usr/bin/env python3
"""
Combine annotations.csv files with the scores that produced the pairs.

Report on any sentinels wrong.
"""

import glob
import csv

for f in glob.glob("data/iterations/*/annotations*.csv"):
    wrong = 0
    for oldf, newf, score in csv.reader(open(f)):
        score = int(score)
        if oldf == newf:
            if score:
                wrong += 1
    print(f"{wrong} wrong in {f}")
