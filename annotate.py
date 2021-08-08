#!/usr/bin/env python3
"""
Simple script for manual annotation.

Type 'a' if there is no difference.
Type 'l' if there is a difference.
Type 'q' to quit.

If your terminal gets corrupted after exit, simply type 'reset'.
"""

# Change this to the name of your command-line audio player
AUDIO_PLAYER = "mpv"

import csv
import curses
import glob
import os.path
import random
import sys
import time

import click


def choose_pair(pairs):
    if len(pairs) == 0:
        print("Done with all annotations.")
        sys.exit(0)

    oldf, newf = pairs[0]
    pairs = pairs[1:]

    if random.random() < 0.5:
        a, b = oldf, newf
    else:
        b, a = oldf, newf
    return oldf, newf, a, b, pairs


def write_rows(goldcsv, goldrows):
    csvout = csv.writer(open(goldcsv, "wt"))
    for row in goldrows:
        csvout.writerow(row)


def check_input(pairs, stdscr, oldf, newf, a, b, goldcsv, goldrows, model_name):
    c = stdscr.getch()
    if c == ord("a"):
        goldrows.append([oldf, newf, 0])
        write_rows(goldcsv, goldrows)
        oldf, newf, a, b, pairs = choose_pair(pairs)
    elif c == ord("l"):
        if oldf != newf:
            goldrows.append([oldf, newf, 1])
            write_rows(goldcsv, goldrows)
        oldf, newf, a, b, pairs = choose_pair(pairs)
    elif c == ord("q"):
        return None, None, None, None, None
    return oldf, newf, a, b, pairs


@click.command()
@click.argument("model_name")
def annotate(model_name):
    goldcsv = f"data/iterations/{model_name}/annotations.csv"
    if os.path.exists(goldcsv):
        goldrows = [row for row in csv.reader(open(goldcsv))]
    else:
        goldrows = []

    pairs = []
    for oldf, newf in csv.reader(open(f"data/iterations/{model_name}/to_label.csv")):
        pairs.append((oldf, newf))

    # https://stackoverflow.com/a/42444516/82733
    stdscr = curses.initscr()
    curses.noecho()
    stdscr.nodelay(1)  # set getch() non-blocking

    stdscr.addstr(0, 0, "a = no difference, l = difference, q = quit")
    line = 1
    oldf, newf, a, b, pairs = choose_pair(pairs)
    try:
        while 1:
            print("Play A")
            os.system(f"{AUDIO_PLAYER} {a} > /dev/null 2>&1")
            oldf, newf, a, b, pairs = check_input(
                pairs, stdscr, oldf, newf, a, b, goldcsv, goldrows, model_name
            )
            if oldf == None:
                break
            time.sleep(0.25)
            oldf, newf, a, b, pairs = check_input(
                pairs, stdscr, oldf, newf, a, b, goldcsv, goldrows, model_name
            )
            if oldf == None:
                break
            print("Play B")
            os.system(f"{AUDIO_PLAYER} {b} > /dev/null 2>&1")
            oldf, newf, a, b, pairs = check_input(
                pairs, stdscr, oldf, newf, a, b, goldcsv, goldrows, model_name
            )
            if oldf == None:
                break
    finally:
        curses.endwin()


if __name__ == "__main__":
    annotate()
