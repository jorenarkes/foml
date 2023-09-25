#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Extract features and learn from them, without saving in between.

__author__ = "Johannes Bjerva and Malvina Nissim"
__credits__ = ["Johannes Bjerva", "Malvina Nissim"]
__license__ = "GPL v3"
__version__ = "0.3 (31/08/2020)"
__maintainer__ = "Mike Zhang"
__email__ = "mikz@itu.dk"
__status__ = "Testing"

import argparse
import logging
import os
import time
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from feature_extractor import (
    features_to_one_hot,
    find_ngrams,
    get_column_types,
    get_line_features,
    preprocess,
    read_features_from_csv,
)
from learn_from_data import (
    baseline,
    evaluate_classifier,
    get_classifiers,
    make_splits,
    read_features,
    show_confusion_matrix,
)

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.DEBUG)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="feature csv filename", type=str, required=True)
    parser.add_argument(
        "--fname", help="filename to store features", type=str, default=None
    )
    parser.add_argument("--nwords", type=int)
    parser.add_argument("--nchars", type=int)
    parser.add_argument(
        "--split",
        help="Indicate what split the ML model has to use",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument("--features", nargs="+", default=[])
    parser.add_argument(
        "--dtype", help="datatype in file", type=str, default=None
    )  # TODO: Not implemented
    parser.add_argument(
        "--delimiter", help="csv delimiter", type=str, default=","
    )  # TODO: Not implemented
    parser.add_argument("--lang", help="data language", type=str, default="english")

    parser.add_argument("--npz", help="feature npz filename", type=str)
    parser.add_argument("--algorithms", help="ml algorithms", nargs="+", required=True)
    parser.add_argument("--plot", help="Show plot", action="store_true")
    parser.add_argument("--cm", help="Show confusion matrix", action="store_true")
    parser.add_argument(
        "--norm", help="Normalise confusion matrix", action="store_true"
    )
    parser.add_argument(
        "--min-samples", help="Min leaf samples in decision tree", type=int, default=1
    )
    parser.add_argument(
        "--max-nodes", help="Max leaf nodes in decision tree", type=int, default=None
    )
    parser.add_argument(
        "--k", help="number of neighbours for k-NN", type=int, default=1
    )
    parser.add_argument(
        "--max-train-size",
        help="maximum number of training instances to look at",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--test", help="shows the result on the test set", action="store_true"
    )

    args = parser.parse_args()

    logging.debug(
        "Reading features...",
    )
    X, y = read_features_from_csv(args)
    logging.debug("Using one hot encoding...")
    X, feature_ids = features_to_one_hot(X)
    (
        train_and_dev_X,
        train_and_dev_y,
        train_X,
        train_y,
        dev_X,
        dev_y,
        test_X,
        test_y,
    ) = make_splits(X, y, args)

    if args.max_train_size:
        train_X: int = train_X[: args.max_train_size]
        train_y: int = train_y[: args.max_train_size]

    logging.info(f"There are {len(train_y)} train samples")
    logging.info(
        f"Classifier uses a {args.split[0]}% train and {args.split[1]}% test split."
    )
    baseline(train_y, dev_y)
    classifiers = get_classifiers(args)

    for clf in classifiers:
        if args.test:
            # takes train and dev as training set.
            train_start_time = time.time()
            clf.fit(train_and_dev_X, train_and_dev_y)
            logging.info(
                "Training time (seconds): "
                + str(round(time.time() - train_start_time, 2)),
            )
            training_result: str = evaluate_classifier(
                clf,
                train_and_dev_X,
                train_and_dev_y,
                args,
                cm_title=f"Confusion Matrix - train set - {clf}",
            )
            logging.info(f"Results on the train set:\n{training_result}\n")
            eval_start_time = time.time()
            test_result: str = evaluate_classifier(
                clf,
                test_X,
                test_y,
                args,
                cm_title=f"Confusion Matrix - test test - {clf}",
            )
            logging.info(
                "Evaluation time (seconds): "
                + str(round(time.time() - eval_start_time, 2)),
            )
            logging.info(f"Results on the test set:\n{test_result}\n")

        else:
            # takes train set and divides it in 70/30 split again.
            train_start_time = time.time()
            clf.fit(train_X, train_y)
            logging.info(
                "Training time (seconds): "
                + str(round(time.time() - train_start_time, 2)),
            )
            training_result: str = evaluate_classifier(
                clf,
                train_X,
                train_y,
                args,
                cm_title=f"Confusion Matrix - train set - {clf}",
            )
            logging.info(f"Results on the train set:\n{training_result}\n")
            eval_start_time = time.time()
            dev_result: str = evaluate_classifier(
                clf, dev_X, dev_y, args, cm_title=f"Confusion Matrix - dev set - {clf}"
            )
            logging.info(
                "Evaluation time (seconds): "
                + str(round(time.time() - eval_start_time, 2)),
            )
            logging.info(f"Results on the dev set:\n{dev_result}\n")

        if isinstance(clf, DecisionTreeClassifier):
            # Plot and save a (part of a) Decision Tree
            plt.figure(figsize=(50, 50))
            tree.plot_tree(clf, filled=True, max_depth=8)
            if not os.path.exists("plot_images"):
                os.makedirs("plot_images")
            plt.savefig(
                "plot_images/" + datetime.now().isoformat() + "-Decision Tree.plot.png"
            )

        if isinstance(clf, MultinomialNB):
            for i in range(0, len(clf.classes_)):
                print(
                    f"Class: {clf.classes_[i]} | Log Prior: {clf.class_log_prior_[i]} | Count: {clf.class_count_[i]}"
                )
