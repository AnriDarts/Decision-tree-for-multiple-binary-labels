# pip install pandas scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import os


def sum_of_binary_labels(binary_labels_for_sum, row):
    # declare total sum of the targeted rows
    total_sum = 0
    # go through each binary label and calculate there total sum
    for param in binary_labels_for_sum:
        total_sum += row[param]

    return total_sum


def universal_decision_tree_for_binary_labels(data_path, data_seperator, binary_label_type, threshold_for,
                                              categorical_columns,
                                              split_number_for_training_testing_sets, out_file_name,
                                              binary_labels_for_sum=None,
                                              max_depth=None):
    # load data
    data = pd.read_csv(data_path, sep=data_seperator)

    # if not single binary label generate it
    if binary_label_type == "multiple_params_sum":
        data["positive_answer"] = data.apply(
            lambda row: 1 if sum_of_binary_labels(binary_labels_for_sum, row) >= threshold_for else 0, axis=1)

        # drop binary labels
        data = data.drop(binary_labels_for_sum, axis=1)

        # one hot encode categorical data
        data = pd.get_dummies(data, columns=categorical_columns)

        # shuffle rows
        data = data.sample(frac=1)

        # split training and testing sets
        train_data = data[:split_number_for_training_testing_sets]
        test_data = data[split_number_for_training_testing_sets:]

        # split data as labels and values
        train_x = train_data.drop(["positive_answer"], axis=1)
        train_y = train_data["positive_answer"]
        test_x = test_data.drop(["positive_answer"], axis=1)
        test_y = test_data["positive_answer"]
        data_x = data.drop(["positive_answer"], axis=1)
        data_y = data["positive_answer"]

        # number of positive_answer in whole dataset
        print("Positive: %d out of %d (%.2f%%)" % (
            np.sum(data_y), len(data_y), 100 * float(np.sum(data_y)) / len(data_y)))

        # if max depth
        if max_depth is not None:
            # fit a decision tree
            t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
            t = t.fit(train_x, train_y)

            # save tree as dot file
            tree.export_graphviz(t, out_file=out_file_name + ".dot", label="all", impurity=False, proportion=True,
                                 feature_names=list(train_x), class_names=["negative", "positive"], filled=True,
                                 rounded=True)

            os.system('dot -Tpng ' + out_file_name + '.dot -o ' + out_file_name + '_graph.png')

            # check score
            scores = cross_val_score(t, data_x, data_y, cv=5)

            # show average score and +/- two standard deviation away
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        else:
            # find optimal number of questions
            i = 0
            depth_acc = np.empty((19, 3), float)
            depth_acc_mean = []
            depth_acc_std = []

            for max_depth in range(1, 20):
                t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
                scores = cross_val_score(t, data_x, data_y, cv=5)
                depth_acc[i, 0] = max_depth
                depth_acc[i, 1] = scores.mean()
                depth_acc[i, 2] = scores.std() * 2

                depth_acc_mean.append(scores.mean())
                depth_acc_std.append(scores.std() * 2)
                i += 1

            sum_array = np.add(depth_acc_mean, depth_acc_std)
            optimal_max_depth = sum_array.argmax() + 1
            print("Optimal max depth for this tree is: " + str(optimal_max_depth))
            # fit a decision tree
            t = tree.DecisionTreeClassifier(criterion="entropy", max_depth=optimal_max_depth)
            t = t.fit(train_x, train_y)

            # save tree as dot file
            tree.export_graphviz(t, out_file=out_file_name + ".dot", label="all", impurity=False, proportion=True,
                                 feature_names=list(train_x), class_names=["positive", "negative"], filled=True,
                                 rounded=True)

            os.system('dot -Tpng ' + out_file_name + '.dot -o ' + out_file_name + '_graph.png')

            # check score
            scores = cross_val_score(t, data_x, data_y, cv=5)

            # show average score and +/- two standard deviation away
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            # create depth graph
            fig, ax = plt.subplots()
            ax.errorbar(depth_acc[:, 0], depth_acc[:, 1], yerr=depth_acc[:, 2])
            plt.savefig(out_file_name + ".png")


universal_decision_tree_for_binary_labels("student-por.csv", ";", "multiple_params_sum", 35,
                                          ["sex", "school", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason",
                                           "guardian",
                                           "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",
                                           "romantic"], 500,
                                          "test", ["G1", "G2", "G3"], 3)
