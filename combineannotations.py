#!/usr/bin/env python3
"""
Combine annotations.csv files with the scores that produced the pairs.

Report on any sentinels wrong.
"""

import glob
import json
import csv
import os.path

MODEL = "openl3gold"

combined = []
for f in glob.glob("data/iterations/*/annotations*.csv"):
    wrong = 0
    files_to_score_label = {}
    for oldf, newf, label in csv.reader(open(f)):
        oldf, newf = os.path.normpath(oldf), os.path.normpath(newf)
        label = int(label)
        if oldf == newf:
            if label:
                wrong += 1
            continue
        oldf2 = oldf.replace(os.path.split(f)[0] + "/", "")
        newf2 = newf.replace(os.path.split(f)[0] + "/", "")
        assert oldf2 != oldf
        assert newf2 != newf
        assert os.path.exists(oldf2), oldf2
        assert os.path.exists(newf2), newf2
        obj = json.loads(open(newf2 + MODEL + ".json").read())
        assert os.path.normpath(obj[0]) == newf2, f"{obj[0]} != {newf2}"
        assert os.path.normpath(obj[1]) == oldf2, f"{obj[1]} != {oldf2}"
        score = obj[2]
        # Remove duplicates
        files_to_score_label[(oldf2, newf2)] = (score, label)
    for oldf2, newf2 in files_to_score_label:
        score, label = files_to_score_label[(oldf2, newf2)]
        combined.append((score, oldf2, newf2, label))
    print(f"{wrong} wrong in {f}")

combined.sort()
combinedcsv = csv.writer(open("data/iterations/combined-annotations.csv", "wt"))
for row in combined:
    combinedcsv.writerow(row)
