#!/usr/bin/env python3
"""
Remove audio that has NSFW tags.
"""

import glob
import json
import os

NSFW = frozenset(
    [
        "cursing",
        "curse",
        "swearing",
        "swear",
        "fuck",
        "fucking",
        "shit",
        "cunt",
        "cock",
        "ass",
        "arse",
        "sex",
        "cumming",
        "sexy",
        "moan",
        "masturbating",
    ]
)

ids = set()
for f in glob.glob("data/orig/FSD50K.metadata/*clips_info_FSD50K.json"):
    obj = json.loads(open(f).read())
    for id in obj:
        matches = set(obj[id]["tags"]) & NSFW
        if matches:
            ids.add(id)

print(f"{len(ids)} ids to remove")
tot = 0
for root, dirs, files in os.walk("data"):
    for filename in files:
        found = False
        for id in ids:
            if filename.startswith(f"{id}.wav"):
                found = True
                break
        if found:
            filename = os.path.join(root, filename)
            os.remove(filename)
            tot += 1
print(f"{tot} files removed")
