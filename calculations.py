import os
import numpy as np
from conlleval import conlleval

#inputs

working_directory = os.getcwd() + "/execution/"
calculations_directory = 'execution_10_Jun_21'
iteration_start = 1
iteration_end = 4

#Functions used

def read_conll(filename):
    raw = open(filename, 'r').readlines()
    all_x = []
    point = []
    for line in raw:
        stripped_line = line.strip().split(' ')
        point.append(stripped_line)
        if line == '\n':
            if len(point[:-1]) > 0:
                all_x.append(point[:-1])
            point = []
    all_x = all_x
    return all_x

def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

os.system('mkdir -p ' + working_directory + '/calculations/')

calculations_write = open(working_directory + '/calculations/calculations_' + calculations_directory + '.txt', "w")

while (iteration_start <= iteration_end):
    calculation_subdir = "iteration" + str(iteration_start) + '/'

    calculations_write.write(calculation_subdir + "\n" + "\n")
    print(calculation_subdir)
    # Change to the working directory and create necessary folder structures
    os.chdir(working_directory + '/' + calculations_directory + '/' + calculation_subdir)
    path = os.getcwd()
    print(path)
    path_new = os.getcwd()
    print(path_new)

    aggregation_train = read_conll('train_sentences_aggregate.txt')
    length_aggregation_train = len(aggregation_train)
    print("Length of all prediction: ", length_aggregation_train)
    words_aggregation_train = [[c[0] for c in x] for x in aggregation_train]
    label_aggregation_train = [[c[1] for c in x] for x in aggregation_train]

    aggregation_test = read_conll('test_sentences_aggregate.txt')
    length_aggregation_test = len(aggregation_test)
    print("Length of all prediction: ", length_aggregation_test)
    words_aggregation_test = [[c[0] for c in x] for x in aggregation_test]
    label_aggregation_test = [[c[1] for c in x] for x in aggregation_test]

    aggregation = read_conll('total_sentences_aggregate.txt')
    length_aggregation = len(aggregation)
    print("Length of all prediction: ", length_aggregation)
    words_aggregation = [[c[0] for c in x] for x in aggregation]
    label_aggregation = [[c[1] for c in x] for x in aggregation]

    ground_truth = read_conll('dataset_sentence_gt.txt')
    length_ground_truth = len(ground_truth)
    print("Length of all prediction: ", length_ground_truth)
    words_groundtruth = [[c[0] for c in x] for x in ground_truth]
    groundtruth_label = [[c[1] for c in x] for x in ground_truth]

    ground_truth_test = read_conll('test_sentences_groundtruth.txt')
    length_ground_truth_test = len(ground_truth_test)
    print("Length of all prediction: ", length_ground_truth_test)
    words_groundtruth_test = [[c[0] for c in x] for x in ground_truth_test]
    groundtruth_label_test = [[c[1] for c in x] for x in ground_truth_test]

    ground_truth_train = read_conll('train_sentences_groundtruth.txt')
    length_ground_truth_train = len(ground_truth_train)
    print("Length of all prediction: ", length_ground_truth_train)
    words_groundtruth_train = [[c[0] for c in x] for x in ground_truth_train]
    groundtruth_label_train = [[c[1] for c in x] for x in ground_truth_train]

    results_total = conlleval(label_aggregation, groundtruth_label, words_groundtruth, 'results_total.txt')
    print("Results for Totalset without cnn:" + str(results_total))

    calculations_write.write("Results for Totalset without BERT: " + str(results_total) + "\n")

    results_train = conlleval(label_aggregation_train, groundtruth_label_train, words_groundtruth_train, 'results_train.txt')
    print("Results for Trainset without BERT:" + str(results_train))

    calculations_write.write("Results for Trainset without BERT: " + str(results_train) + "\n")

    results_test = conlleval(label_aggregation_test, groundtruth_label_test, words_groundtruth_test, 'results_test.txt')
    print("Results for Testset without BERT:" + str(results_test))

    calculations_write.write("Results for Testset without BERT: " + str(results_test) + "\n")

    aggregation_train = read_conll('train_sentences_aggregate_cnn.txt')
    length_aggregation_train = len(aggregation_train)
    print("Length of all prediction: ", length_aggregation_train)
    words_aggregation_train = [[c[0] for c in x] for x in aggregation_train]
    label_aggregation_train = [[c[1] for c in x] for x in aggregation_train]


    aggregation_test = read_conll('test_sentences_aggregate_cnn.txt')
    length_aggregation_test = len(aggregation_test)
    print("Length of all prediction: ", length_aggregation_test)
    words_aggregation_test = [[c[0] for c in x] for x in aggregation_test]
    label_aggregation_test = [[c[1] for c in x] for x in aggregation_test]

    aggregation = read_conll('total_sentences_aggregate_cnn.txt')
    length_aggregation = len(aggregation)
    print("Length of all prediction: ", length_aggregation)
    words_aggregation = [[c[0] for c in x] for x in aggregation]
    label_aggregation = [[c[1] for c in x] for x in aggregation]

    results_total = conlleval(label_aggregation, groundtruth_label, words_groundtruth, 'results_total_cnn.txt')
    print("Results for Totalset with BERT:" + str(results_total))

    calculations_write.write("Results for Totalset with BERT: " + str(results_total) + "\n")

    results_train = conlleval(label_aggregation_train, groundtruth_label_train, words_groundtruth_train, 'results_train_cnn.txt')
    print("Results for Trainset with BERT:" + str(results_train))

    calculations_write.write("Results for Trainset with BERT: " + str(results_train) + "\n")

    results_test = conlleval(label_aggregation_test, groundtruth_label_test, words_groundtruth_test, 'results_test_cnn.txt')
    print("Results for Testset with BERT:" + str(results_test))

    calculations_write.write("Results for Testset with BERT: " + str(results_test) + "\n")
    calculations_write.write("\n")

    iteration_start = iteration_start + 1

calculations_write.close()