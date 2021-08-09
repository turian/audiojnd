#!/usr/bin/env python3
"""
Given model scores, for each transformation type, select K
transformations that greedily maximize the all-pairs distance between
the distance of the K transforms versus the original. (Maximum dispersion.)

The simplest way to do this heuristically is kmeans.
"""

import csv
import glob
import json
import os.path
import random
import shutil
from collections import Counter, defaultdict

import click
import numpy as np
import sklearn.cluster
import sklearn.metrics
from tqdm.auto import tqdm

# N = 2**3 - 1
# N = 2**1 + 1
N = 10

"""
def select_transforms(transform_to_distance, minscore, maxscore, N=N-2):
    #Since we do a greedy binary search to cover the space,
    #N should be 2**k-1 (excluding the min and max).
    if N == 0:
        return []
    print(minscore, maxscore)
    meanscore = (minscore + maxscore) / 2
    dists = np.array([t[0] for t in transform_to_distance])
    argmin = np.argmin(np.abs(dists - meanscore))
#    return transform_to_distance[np.argmin(np.abs(dists - meanscore))] +
#        select_transforms(transform_to_distance, minscore, maxscore, N=N-2)
"""


@click.command()
@click.argument("model_name")
def selecttransforms(model_name):
    base_dir = os.path.join("data/iterations", model_name)
    os.makedirs(base_dir)
    files = list(glob.glob(f"data/transforms/*/*/*.mp3{model_name}.json"))
    random.seed(0)
    random.shuffle(files)
    transform_to_distance = defaultdict(list)
    for f in tqdm(files):
        try:
            transformjson = f.replace(f".mp3{model_name}", "")
            transform = json.loads(open(transformjson).read())
            transform_type = list(transform[1].keys())[0]
            transform_file, original_file, distance = json.loads(open(f).read())
            assert transform_file.startswith(
                "data/transforms/"
            ) or transform_file.startswith(
                "./data/transforms/"
            ), f"{f} {transform_file}, {original_file}"
            transform_to_distance[transform_type].append(
                (distance, transform_file, original_file)
            )
        except:
            print(f"Skipping {f}")

    tot = 0
    to_label = []
    for transform_type in transform_to_distance:
        if len(transform_to_distance[transform_type]) < N:
            continue
        transform_to_distance[transform_type].sort()
        # print(
        #    transform_type,
        #    len(transform_to_distance[transform_type]),
        #    transform_to_distance[transform_type],
        # )
        clf = sklearn.cluster.KMeans(n_clusters=N, random_state=42)
        # KMeans has slightly better dispersion
        # clf = sklearn.cluster.AgglomerativeClustering(n_clusters=N, linkage='average')
        X = np.array([[t[0]] for t in transform_to_distance[transform_type]])
        clf.fit(X)

        cluster_centers_ = [np.mean(X[np.where(clf.labels_ == i)[0]]) for i in range(N)]
        centers = sorted([c for c in cluster_centers_])
        # print(Counter([clf.cluster_centers_[i][0] for i in clf.labels_.tolist()]))

        Z = [[c] for c in cluster_centers_]
        best_idx = sklearn.metrics.pairwise_distances_argmin(Z, X)
        d = np.array(X)[best_idx]
        best_points = sorted([x[0] for x in d.tolist()])
        print(best_points)

        # TODO: Save this to a JSON file
        final_scores = sorted(
            np.array(transform_to_distance[transform_type])[best_idx].tolist()
        )
        open(os.path.join(base_dir, f"{transform_type}.json"), "wt").write(
            json.dumps(final_scores, indent=4)
        )

        def copyfilewithpath(src):
            newpath = os.path.join(base_dir, src)
            newdir = os.path.split(newpath)[0]
            if not os.path.exists(newdir):
                os.makedirs(newdir)
            shutil.copy(src, newpath)

        for score, transformed_file, orig_file in final_scores:
            print(score, transformed_file, orig_file)
            print(list(glob.glob(os.path.splitext(transformed_file)[0] + "*")))
            copyfilewithpath(transformed_file)
            copyfilewithpath(orig_file)
            to_label.append(
                [
                    os.path.join(base_dir, orig_file),
                    os.path.join(base_dir, transformed_file),
                ]
            )

        # print("l1 dispersion", np.sum(sklearn.metrics.pairwise.euclidean_distances(d)))
        # print("l1 dispersion", np.sum(sklearn.metrics.pairwise.manhattan_distances(d)))
        # tot += np.sum(sklearn.metrics.pairwise.manhattan_distances(d))
    # print(tot)
    random.shuffle(to_label)
    to_label_csv = csv.writer(open(os.path.join(base_dir, "to_label.csv"), "wt"))
    for row in to_label:
        to_label_csv.writerow(row)
    to_label_csv = None


if __name__ == "__main__":
    selecttransforms()
