#All required packages are loaded
import os
import shutil
import math
import os
import numpy as np
from datetime import datetime
from bert import Ner
from compute_class_weights import outcome

# Functions

def array_to_sentence(array):
    sentences = ' '.join(array)
    return sentences

def confidence_measurement_word(sentence, word):
    a = sentence
    b = word
    aggregate_prob_array = []
    for j in range(0, len(ystar_label_aggregate[a][b])):
        if ystar_label_aggregate[a][b][j] > 0:
            aggregate_prob_array.append(ystar_label_aggregate[a][b][j])
    minimum = aggregate_prob_array[0]
    maximum = aggregate_prob_array[0]
    if len(aggregate_prob_array) == 1:
        confidence_measure_word = 1
    else:
        for k in range(0, len(aggregate_prob_array)):
            value = aggregate_prob_array[k]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        confidence_measure_word = maximum-minimum
    return confidence_measure_word

def confidence_measurement_word_cnn(sentence, word):
    a = sentence
    b = word
    aggregate_prob_array = []
    for j in range(0, len(cnn_test_ystar_label_aggregate[a][b])):
        if cnn_test_ystar_label_aggregate[a][b][j] > 0:
            aggregate_prob_array.append(cnn_test_ystar_label_aggregate[a][b][j])
    minimum = aggregate_prob_array[0]
    maximum = aggregate_prob_array[0]
    if len(aggregate_prob_array) == 1:
        confidence_measure_word = 1
    else:
        for k in range(0, len(aggregate_prob_array)):
            value = aggregate_prob_array[k]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
        confidence_measure_word = maximum-minimum
    return confidence_measure_word

def sum_of_distance_cnn():
    sum_of_distance = 0
    for j in range(0,len(cnn_x_words)):
        distance_for_sentence = 0
        for i in range(0,len(cnn_x_labels[j])):
            distance_for_sentence = distance_for_sentence + (distance_between_labels(cnn_test_ystarcap_label_encoding[j][i],cnn_test_ystar_label_aggregate[j][i]))
        sum_of_distance = sum_of_distance + (distance_for_sentence*cnn_confidence_measurement[j])
    return sum_of_distance

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
def encode(x, n):
    result = np.zeros(n)
    result[x] = 1
    return result
def score(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def onehot_vector_encoding(label):
    test = label2ind.get(label, -1)
    if test != -1:
        one_hot_vector = encode(test-1,len(labels))
    else:
        one_hot_vector = np.zeros(len(labels))
    return one_hot_vector

def multiply_by_worker_weight_one(answers):
    multiply_answers = []
    priority_array = priority_array_input
    Labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    for i in range(0,len(answers)):
        if np.array_equal(answers[i],onehot_vector_encoding(Labels[0])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[0]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[1])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[1]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[2])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[2]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[3])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[3]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[4])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[4]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[5])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[5]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[6])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[6]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[7])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[7]
        elif np.array_equal(answers[i],onehot_vector_encoding(Labels[8])):
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[8]
        else:
            multiply_answers_test = (answers[i]*worker_weights[i])*priority_array[8]
        multiply_answers.append(multiply_answers_test)
    return multiply_answers

def ystar_label_one(answers):
    answers_updated = multiply_by_worker_weight_one(answers)
    ystar_label = sum(answers_updated)/np.sum(sum(answers_updated))
    return ystar_label

def multiply_by_worker_weight(answers, ystarcapanswer):
    multiply_answers = []
    priority_array = priority_array_input
    for i in range(0, len(answers)):
        if np.array_equal(answers[i], onehot_vector_encoding(Labels[0])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[0]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[1])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[1]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[2])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[2]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[3])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[3]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[4])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[4]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[5])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[5]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[6])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[6]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[7])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[7]
        elif np.array_equal(answers[i], onehot_vector_encoding(Labels[8])):
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[8]
        else:
            multiply_answers_test = (answers[i] * worker_weights[i]) * priority_array[8]
        multiply_answers.append(multiply_answers_test)
    if np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[0])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[0]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[1])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[1]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[2])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[2]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[3])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[3]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[4])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[4]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[5])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[5]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[6])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[6]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[7])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[7]
    elif np.array_equal(ystarcapanswer, onehot_vector_encoding(Labels[8])):
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[8]
    else:
        multiply_answers_test_ystarcap = (ystarcapanswer * worker_weight_cnn[0]) * priority_array[8]
    multiply_answers.append(multiply_answers_test_ystarcap)
    return multiply_answers


def ystar_label(answers, ystarcapanswer):
    answers_updated = multiply_by_worker_weight(answers, ystarcapanswer)
    ystar_label = sum(answers_updated) / np.sum(sum(answers_updated))
    return ystar_label


def distance_between_labels(label1, label2):
    if len(label1) > 0:
        label2 = np.array(label2)
        one_hot = np.array(onehot_vector_encoding('O'))
        if np.array_equal(label2, one_hot):
            return 0
        else:
            ranger = -1
            for i in range(0, len(label1)):
                if label1[i] > 0:
                    ranger = i
            if ranger == -1:
                distance = 1
            else:
                value = label2[ranger] + 0.00001
                if value < 0:
                    value = -value
                elif value == 0:
                    value = value + 0.00001
                print(value)
                distance = -math.log(value, 2)
            return distance
    else:
        return 1

def no_of_annotations(worker_number):
    a = worker_number
    number_of_annotations = 0
    for i in range(0, len(ystar_x_words)):
        k = 0
        if ystar_x_labels[i][k][a] != '?':
            number_of_annotations = number_of_annotations + int(number_of_annotations_sentence[i][0])
    return number_of_annotations

def sentences_worked_by_worker(worker_number):
    a = worker_number
    sentence_list_worked_by_worker = []
    for i in range(0,len(ystar_x_words)):
        k=0
        if ystar_x_labels[i][k][a] !='?':
            sentence_list_worked_by_worker.append(i)
    return sentence_list_worked_by_worker

def sum_of_distance(worker_number):
    a = worker_number
    sum_of_distance = 0
    x = []
    x = sentences_worked_by_worker(a)
    for m in range(0,len(x)):
        z = x[m]
        distance_for_sentence = 0
        for i in range(0,len(ystar_x_labels[z])):
            distance_for_sentence = distance_for_sentence + (distance_between_labels(ystar_test_label_encoding[z][i][a],ystar_label_aggregate[z][i]))
        sum_of_distance = sum_of_distance + (distance_for_sentence*confidence_measurement_data[z])
    return sum_of_distance
def sum_of_confidence_measure(worker_number):
    a = worker_number
    sum_of_confidence_measure = 0
    x = []
    x = sentences_worked_by_worker(a)
    for m in range(0,len(x)):
        z = x[m]
        sum_of_confidence_measure = sum_of_confidence_measure + confidence_measurement_data[z]
    return sum_of_confidence_measure

def minimum_label(index_array, sentence_number, word_number):
    a = sentence_number
    i = word_number
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    minimum = index_array[0]
    for k in range(0, len(index_array)):
        index = index_array[k]
        label_value = -math.log(ystar_label_aggregate[a][i][index], 2)
        min_value = -math.log(ystar_label_aggregate[a][i][minimum], 2)
        if label_value < min_value:
            minimum = index_array[k]
    min_label_aggregate_one_hot = encode(minimum,len(labels))
    for k in range(0,len(actual_label_one_hot)):
        if np.array_equal(actual_label_one_hot[k],min_label_aggregate_one_hot):
            min_label_aggregate = labels[k]
    return min_label_aggregate, minimum

def confidence_measurement(sentence_number):
    a = sentence_number
    sentence_length = len(ystar_label_aggregate[a])
    confidence_measure_sentence = 0
    for i in range(0, len(ystar_label_aggregate[a])):
        aggregate_prob_array = []
        for j in range(0, len(ystar_label_aggregate[a][i])):
            if ystar_label_aggregate[a][i][j] > 0:
                aggregate_prob_array.append(ystar_label_aggregate[a][i][j])
        minimum = aggregate_prob_array[0]
        maximum = aggregate_prob_array[0]
        if len(aggregate_prob_array) == 1:
            confidence_measure_word = 1
        else:
            for k in range(0, len(aggregate_prob_array)):
                value = aggregate_prob_array[k]
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
            confidence_measure_word = maximum - minimum

        confidence_measure_sentence = confidence_measure_sentence + confidence_measure_word
    confidence_measurement = confidence_measure_sentence / sentence_length
    return confidence_measurement


transition_probability_forward = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

transition_probability_backward = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

def calculate_transition_weight(previous_label, current_label, next_label):
    transition_weight = [1, 1.1, 1.1]
    return_weight = 0
    if next_label == -1 :
        if previous_label == 9:
            return_weight = transition_weight[1]
        else:
            if previous_label == current_label:
                return_weight = transition_weight[1]
            elif (previous_label == 8) and (current_label != 8):
                return_weight = transition_weight[2]
            elif (previous_label != 8) and (current_label == 8):
                return_weight = transition_weight[0]
    elif previous_label == -1:
        if next_label == 9:
            return_weight = transition_weight[1]
        else:
            if next_label == current_label:
                return_weight = transition_weight[1]
            elif (current_label != 8) and (next_label == 8):
                return_weight = transition_weight[2]
            elif (current_label == 8) and (next_label != 8):
                return_weight = transition_weight[0]

    return return_weight


def best_labels_forward(probability_labels, transition_probability_forward):
    previous_label = 9
    best_labels = []
    for i in range(0, len(probability_labels)):
        updated_probabilities = []
        for j in range(0, len(probability_labels[i])):
            updated_probability = probability_labels[i][j]*transition_probability_forward[j][previous_label]
            updated_probabilities.append(updated_probability)
        max_value = max(updated_probabilities)
        best_label = updated_probabilities.index(max_value)
        best_labels.append(best_label)
        previous_label = best_label
    return best_labels

def best_labels_backward(probability_labels, transition_probability_backward):
    next_label = 9
    best_labels = []
    for i in range(len(probability_labels)-1, -1, -1):
        updated_probabilities = []
        for j in range(0, len(probability_labels[i])):
            updated_probability = probability_labels[i][j]*transition_probability_backward[j][next_label]
            updated_probabilities.append(updated_probability)
        max_value = max(updated_probabilities)
        best_label = updated_probabilities.index(max_value)
        best_labels.append(best_label)
        next_label = best_label
    print(best_labels)
    best_labels.reverse()
    return best_labels

def viterbi_sequence_labels(sentence_number):
    probability_labels = ystar_label_aggregate[sentence_number]
    best_labels_for = best_labels_forward(probability_labels, transition_probability_forward)
    best_labels_back = best_labels_backward(probability_labels, transition_probability_backward)
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    sequence_labels_forward = []
    entity_count_forward = 0
    for i in range(0, len(best_labels_for)):
        index = best_labels_for[i]
        if index < 8:
            entity_count_forward = entity_count_forward + 1
        sequence_labels_forward.append(labels[index])

    sequence_labels_backward = []
    entity_count_backward = 0
    for i in range(0, len(best_labels_back)):
        index = best_labels_back[i]
        if index < 8:
            entity_count_backward = entity_count_backward + 1
        sequence_labels_backward.append(labels[index])
    sequence_labels = sequence_labels_backward
    if entity_count_forward > entity_count_backward:
        sequence_labels = sequence_labels_forward

    return sequence_labels

#Provide the working directory, this is the location where this script is located.
working_directory = os.getcwd() + "/"
pre_processing_iteration = "execution/iteration0"
bert_pretrained_model_path = working_directory + "/pre_trained_bert/"
priority_array_input = [0.24089703400048226, 0.29920171937365675, 0.4449191900427271, 0.20218214206190158, 0.23652694610778444, 0.28697075066475763, 0.34636080430472954, 0.18147625763986835, 0.10918054470029459]
worker_annotators_count = 47
weight_array = [0.7]*47
weight_array_init = [0]*47

weight_vector_prev_iteration = np.asarray(weight_array_init)
weight_vector_this_iteration = np.asarray(weight_array)

threshold_difference = 0.1

priority_array_file = open(working_directory + pre_processing_iteration + "/priority_array_file.txt", "a")

#Execution begins from here
datetimeobj = datetime.now()
date = datetimeobj.strftime("%d_%b_%y")
execution_dir = "execution/execution_" + date + '/'
iteration = 0
while((abs((np.sum(np.subtract(weight_vector_this_iteration,weight_vector_prev_iteration))))) > threshold_difference):
    weight_vector_prev_iteration = weight_vector_this_iteration
    iteration = iteration + 1
    iteration_dir = "iteration" + str(iteration) + '/'
    prev_iteration_dir = "iteration" + str(iteration-1) + '/'
    #Change to the working directory and create necessary folder structures
    os.chdir(working_directory)
    path = os.getcwd()
    print(path)
    os.system('mkdir -p ' + execution_dir + '/' + iteration_dir)
    os.chdir(execution_dir + '/' + iteration_dir)
    path_new = os.getcwd()
    print(path_new)
    DATA_PATH = working_directory + execution_dir + '/' + iteration_dir + '/'
    #Create necessary folder structure for pre-processing
    os.system('mkdir -p data')
    os.system('mkdir -p dictionary_directory/dictionary_sentence_list')
    os.system('mkdir -p dictionary_directory/dictionary_aggregate_sentence_list')
    os.system('mkdir -p dictionary_directory/dictionary_gt_sentence_list')
    os.system('mkdir individual_files')
    os.system('mkdir -p dictionary_directory/dictionary_worker_sentence')
    os.system('mkdir -p dictionary_directory/dictionary_sentence_worker')
    os.system('mkdir -p dictionary_directory/dictionary_aggregate_'+ iteration_dir + '_sentence_list')


    # Copy required files for processing
    conllevalpl = shutil.copy("../../../conlleval.pl", "conlleval.pl")
    conllevalpy = shutil.copy("../../../conlleval.py", "conlleval.py")
    conllevalpyc = shutil.copy("../../../conlleval.pyc", "conlleval.pyc")
    answers = shutil.copy("../../../" + pre_processing_iteration + "/data/answers.txt", "data/answers.txt")
    groundtruth = shutil.copy("../../../" + pre_processing_iteration + "/data/Ground_Truth.txt", "data/Ground_Truth.txt")
    number_of_annotations = shutil.copy("../../../" + pre_processing_iteration + "/individual_files/number_of_annotations.txt", "data/number_of_annotations.txt")
    word_index = shutil.copy("../../../" + pre_processing_iteration + "/individual_files/word_index.txt", "data/word_index.txt")
    dataset_answers = shutil.copy("../../../" + pre_processing_iteration + "/data/answers.txt", "dataset_sentence_complete.txt")
    dataset_ground_truth = shutil.copy("../../../" + pre_processing_iteration + "/data/Ground_Truth.txt", "dataset_sentence_gt.txt")

    if(iteration==1):
        aggregate_label = shutil.copy("../../../" + pre_processing_iteration + "/data/mv.txt", "data/aggregation.txt")
        dataset_aggregate_label = shutil.copy("../../../" + pre_processing_iteration + "/data/mv.txt", "dataset_sentences_aggregate.txt")
        worker_weight = shutil.copy("../../../" + pre_processing_iteration + "/data/worker_weight.txt", "data/worker_weight.txt")
        worker_weight_cnn = shutil.copy("../../../" + pre_processing_iteration + "/data/r_theta.txt", "data/r_theta.txt")
    else:
        aggregate_label = shutil.copy("../" + prev_iteration_dir + "/total_sentences_aggregate_cnn.txt", "data/aggregation.txt")
        dataset_aggregate_label = shutil.copy("../" + prev_iteration_dir + "/total_sentences_aggregate_cnn.txt", "dataset_sentences_aggregate.txt")
        worker_weight = shutil.copy("../" + prev_iteration_dir + "/updated_worker_weight.txt", "data/worker_weight.txt")
        worker_weight_cnn = shutil.copy("../" + prev_iteration_dir + "/updated_rtheta.txt", "data/r_theta.txt")


    #Load worker_weights and cnn_weights

    worker_weights = []
    worker_weight_file = open("data/worker_weight.txt", "r")

    for line in worker_weight_file:
        line1 = line.rstrip()
        worker_weights.append(float(line1))
    worker_weight_file.close()
    print(worker_weights)
    print(len(worker_weights))

    worker_weight_cnn = []
    worker_weight_cnn_file = open("data/r_theta.txt", "r")
    for line in worker_weight_cnn_file:
        line1 = line.rstrip()
        worker_weight_cnn.append(float(line1))
    worker_weight_cnn_file.close()
    print(worker_weight_cnn)
    print(len(worker_weight_cnn))

    #Load number of annotations to consider
    annotations = read_conll('data/number_of_annotations.txt')
    number_of_annotations_sentence = [[c[1] for c in x] for x in annotations]
    print(int(number_of_annotations_sentence[0][0]))

    all_answers = read_conll(DATA_PATH + 'dataset_sentence_complete.txt')
    all_mv = read_conll(DATA_PATH + 'dataset_sentences_aggregate.txt')
    all_test = read_conll(DATA_PATH + 'dataset_sentence_gt.txt')
    all_docs = all_test
    length_all_answers = len(all_answers)
    length_all_mv = len(all_mv)
    length_all_test = len(all_test)
    length_all_docs = len(all_docs)
    print("Answers data size:" + str(length_all_answers))
    print("Majority voting data size:" + str(length_all_mv))
    print("Test data size:" + str(length_all_test))
    print("Total sequences:" + str(length_all_docs))

    X_train = [[c[0] for c in x] for x in all_answers]
    y_answers = [[c[1:] for c in y] for y in all_answers]
    y_mv = [[c[1] for c in y] for y in all_mv]
    X_test = [[c[0] for c in x] for x in all_test]
    y_test = [[c[1] for c in y] for y in all_test]
    X_all = [[c[0] for c in x] for x in all_docs]
    y_all = [[c[1] for c in y] for y in all_docs]

    aggregate_read = read_conll("dataset_sentences_aggregate.txt")
    X_aggregate_read = [[c[0] for c in x] for x in aggregate_read]
    Y_aggregate_read = [[c[1] for c in x] for x in aggregate_read]
    y_weights = y_answers
    for i in range(0, len(Y_aggregate_read)):
        for j in range(0, len(Y_aggregate_read[i])):
            y_weights[i][j].append(Y_aggregate_read[i][j])

    print(len(y_weights[0][0]))
    priority_array_input = outcome(X_train, y_weights)
    entity_value = np.sum(priority_array_input) / len(priority_array_input)
    priority_array_input_new = []
    for i in range(0, 8):
        priority_array_input_new.append(entity_value)
    priority_array_input_new.append(1)
    priority_array_input = priority_array_input_new
    print(priority_array_input)
    priority_array_file.write(str(priority_array_input))
    priority_array_file.write("\n")

    N_ANNOT = len(y_answers[0][0])
    print("Num annnotators:" + str(N_ANNOT))

    lengths = [len(x) for x in all_docs]
    all_text = [c for x in X_all for c in x]
    words = list(set(all_text))
    word2ind = {word: index for index, word in enumerate(words)}
    ind2word = {index: word for index, word in enumerate(words)}
    labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    print("Labels:" + str(labels))
    label2ind = {label: (index + 1) for index, label in enumerate(labels)}
    ind2label = {(index + 1): label for index, label in enumerate(labels)}
    ind2label[0] = "P"
    max_length = max(lengths)
    min_length = min(lengths)
    print("Input sequence length range: " + " Max " + str(max_length) + " Min " + str(min_length))

    max_label = max(label2ind.values()) + 1
    print("Max label:" + str(max_label))

    maxlen = max([len(x) for x in X_all])
    print("Maximum sequence length:" + str(maxlen))

    Labels = ['B-LOC', 'B-PER', 'B-ORG', 'B-MISC', 'I-LOC', 'I-PER', 'I-ORG', 'I-MISC', 'O']
    labels_one_hot = []
    for i in range(0, len(Labels)):
        label_one_hot = onehot_vector_encoding(Labels[i])
        labels_one_hot.append(label_one_hot)
    print(labels_one_hot)

    actual_label_one_hot = []
    for i in range(len(labels)):
        actual_label_one_hot_i = onehot_vector_encoding(labels[i])
        actual_label_one_hot.append(actual_label_one_hot_i)
    print(actual_label_one_hot)

    ystar_x_answers = read_conll(DATA_PATH + 'dataset_sentence_complete.txt')
    ystar_x_words = [[c[0] for c in x] for x in ystar_x_answers]
    ystar_x_labels = [[c[1:] for c in y] for y in ystar_x_answers]
    print(ystar_x_labels[0][0])

    if (iteration == 1):
        ystar_test_label_encoding = []
        for i in range(0, len(ystar_x_labels)):
            ystar_test_label_i = []
            for j in range(0, len(ystar_x_labels[i])):
                ystar_test_label_j = []
                for k in range(0, len(ystar_x_labels[i][j])):
                    label_y_one_hot = onehot_vector_encoding(ystar_x_labels[i][j][k])
                    ystar_test_label_j.append(label_y_one_hot)
                ystar_test_label_i.append(ystar_test_label_j)
            ystar_test_label_encoding.append(ystar_test_label_i)

        print(ystar_test_label_encoding[0][0])

    ystar_label_aggregate = []
    for i in range(0, len(ystar_test_label_encoding)):
        ystar_label_aggregate_i = []
        for j in range(0, len(ystar_test_label_encoding[i])):
            label_ystar_aggregate = ystar_label_one(ystar_test_label_encoding[i][j])
            ystar_label_aggregate_i.append(label_ystar_aggregate)
        ystar_label_aggregate.append(ystar_label_aggregate_i)
    print(ystar_label_aggregate[0])

    confidence_measurement_data = []
    for i in range(0, len(ystar_label_aggregate)):
        confidence_measurement_data.append(confidence_measurement(i))
    print(len(confidence_measurement_data))
    print(confidence_measurement_data[0])

    test_sentences = []
    train_sentences = []
    for i in range(0, len(confidence_measurement_data)):
        if confidence_measurement_data[i] <= 0.95:
            test_sentences.append(i)
        elif confidence_measurement_data[i] > 0.95:
            train_sentences.append(i)
    print(len(test_sentences))
    print(len(train_sentences))
    print(len(test_sentences) + len(train_sentences))

    # Finding updated worker weights for all workers

    if (iteration > 0):
        distance_workers = []
        for i in range(0, worker_annotators_count):
            sentences_by_worker = sentences_worked_by_worker(i)
            if len(sentences_by_worker) != 0:
                distance_workers.append(no_of_annotations(i)/sum_of_distance(i))
            else:
                distance_workers.append(1000)
        max_of_worker_distances = np.max(distance_workers) + 1e-7
        for i in range(0, worker_annotators_count):
            worker_weights[i] = -math.log(distance_workers[i] / max_of_worker_distances)

    # Finding aggregate labels for all the sentences
    Label_sentence_agg = []
    for i in range(0, len(ystar_x_words)):
        Label_sentence = viterbi_sequence_labels(i)
        Label_sentence_agg.append(Label_sentence)
    print(Label_sentence_agg)

    Label_aggregate_testing = []
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        Label_aggregate_testing.append(Label_sentence_agg[sentence_number])

    Label_aggregate_train = []
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        Label_aggregate_train.append(Label_sentence_agg[sentence_number])

    # Data for model iteration
    total_sentences_aggregate = open("total_sentences_aggregate.txt", "a")
    for i in range(0, len(Label_sentence_agg)):
        for j in range(0, len(Label_sentence_agg[i])):
            total_sentences_aggregate.write(str(ystar_x_words[i][j]))
            total_sentences_aggregate.write(" ")
            total_sentences_aggregate.write(str(Label_sentence_agg[i][j]))
            total_sentences_aggregate.write("\n")
        total_sentences_aggregate.write("\n")
    total_sentences_aggregate.close()

    # Data for model iteration
    train_sentences_aggregate = open("train_sentences_aggregate.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            train_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            train_sentences_aggregate.write(" ")
            train_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            train_sentences_aggregate.write("\n")
        train_sentences_aggregate.write("\n")
    train_sentences_aggregate.close()

    # Data for model iteration
    train_sentences_complete = open("train_sentences_complete.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            train_sentences_complete.write(str(ystar_x_words[sentence_number][j]))
            train_sentences_complete.write(" ")
            for k in range(0, len(ystar_x_labels[sentence_number][j])):
                train_sentences_complete.write(str(ystar_x_labels[sentence_number][j][k]))
                train_sentences_complete.write(" ")
            train_sentences_complete.write("\n")
        train_sentences_complete.write("\n")
    train_sentences_complete.close()

    # Data for model iteration
    test_sentences_aggregate = open("test_sentences_aggregate.txt", "a")
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            test_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            test_sentences_aggregate.write(" ")
            test_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            test_sentences_aggregate.write("\n")
        test_sentences_aggregate.write("\n")
    test_sentences_aggregate.close()

    confidence_measure_file = open("confidence_measure_file.txt", "a")
    for i in range(0, len(confidence_measurement_data)):
        confidence_measure_file.write(str(confidence_measurement_data[i]))
        confidence_measure_file.write("\n")
    confidence_measure_file.close()

    gt_answers = read_conll(DATA_PATH + 'dataset_sentence_gt.txt')
    gt_words = [[c[0] for c in x] for x in gt_answers]
    gt_labels = [[c[1] for c in y] for y in gt_answers]
    print(ystar_x_labels[0][0])

    # Data for model iteration
    train_sentences_groundtruth = open("train_sentences_groundtruth.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(gt_words[sentence_number])):
            train_sentences_groundtruth.write(str(gt_words[sentence_number][j]))
            train_sentences_groundtruth.write(" ")
            train_sentences_groundtruth.write(str(gt_labels[sentence_number][j]))
            train_sentences_groundtruth.write("\n")
        train_sentences_groundtruth.write("\n")
    train_sentences_groundtruth.close()

    # Data for model iteration
    test_sentences_groundtruth = open("test_sentences_groundtruth.txt", "a")
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        for j in range(0, len(gt_words[sentence_number])):
            test_sentences_groundtruth.write(str(gt_words[sentence_number][j]))
            test_sentences_groundtruth.write(" ")
            test_sentences_groundtruth.write(str(gt_labels[sentence_number][j]))
            test_sentences_groundtruth.write("\n")
        test_sentences_groundtruth.write("\n")
    test_sentences_groundtruth.close()

    model = Ner(bert_pretrained_model_path)

    all_test = read_conll("test_sentences_aggregate.txt")

    length_all_test = len(all_test)
    print("Test data size:" + str(length_all_test))

    X_test = [[c[0] for c in x] for x in all_test]
    y_test = [[c[1] for c in y] for y in all_test]

    cnn_predict_output_formatted = open("cnn_predict_output_formatted.txt", "a")

    for i in range(0, len(X_test)):
        output = model.predict(array_to_sentence(X_test[i]))
        for j in range(0, len(X_test[i])):
            cnn_predict_output_formatted.write(output[j].get("word"))
            cnn_predict_output_formatted.write(" ")
            cnn_predict_output_formatted.write(output[j].get("tag"))
            cnn_predict_output_formatted.write('\n')
        cnn_predict_output_formatted.write('\n')
    cnn_predict_output_formatted.close()

    sentence_name_testing = []

    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        sentence_name = "sentence_" + str(sentence_number + 1) + ".txt"
        sentence_name_testing.append(sentence_name)
    print(len(sentence_name_testing))

    sentence_name_training = []

    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        sentence_name = "sentence_" + str(sentence_number + 1) + ".txt"
        sentence_name_training.append(sentence_name)
    print(len(sentence_name_training))

    sentence_list = []
    for i in range(0, len(Label_sentence_agg)):
        sentence_number = i
        sentence_name = "sentence_" + str(sentence_number + 1) + ".txt"
        sentence_list.append(sentence_name)
    print(len(sentence_list))

    aggregate_label = open("train_sentences_aggregate.txt", "r")
    counter = 0
    for line in aggregate_label:
        if line not in ['\n', '\r\n']:
            sentence_count = open('dictionary_directory/dictionary_aggregate_'+ iteration_dir + '_sentence_list/' + str(sentence_name_training[counter]), "a")
            sentence_count.write(line)
            sentence_count.close()
        if line in ['\n', '\r\n']:
            counter = counter + 1
    aggregate_label.close()

    aggregate_label = open("cnn_predict_output_formatted.txt", "r")
    counter = 0
    for line in aggregate_label:
        if line not in ['\n', '\r\n']:
            sentence_count = open("dictionary_directory/dictionary_aggregate_"+ iteration_dir + "_sentence_list/" + str(sentence_name_testing[counter]), "a")
            sentence_count.write(line)
            sentence_count.close()
        if line in ['\n', '\r\n']:
            counter = counter + 1
    aggregate_label.close()

    # Data for model iteration
    ystar_aggregate_sentence_complete = open("ystarcap_aggregate.txt", "a")
    for i in range(0, len(sentence_list)):
        sentence_file = sentence_list[i]
        sentence_loop = open("dictionary_directory/dictionary_aggregate_"+ iteration_dir + "_sentence_list/" + str(sentence_file) + "", "r")
        for line1 in sentence_loop:
            ystar_aggregate_sentence_complete.write(str(line1))
        ystar_aggregate_sentence_complete.write("\n")
        sentence_loop.close()
    ystar_aggregate_sentence_complete.close()

    # Loading words, ystar and ystarcap from r_test_new.txt
    ystarcap_prediction = read_conll('ystarcap_aggregate.txt')
    length_ystarcap_prediction = len(ystarcap_prediction)
    print("Length of all prediction: ", length_ystarcap_prediction)
    words_ystarcap = [[c[0] for c in x] for x in ystarcap_prediction]
    ystarcap = [[c[1] for c in x] for x in ystarcap_prediction]
    print(str(words_ystarcap[0]))
    print(str(ystarcap[0]))

    aggregate_read = read_conll("ystarcap_aggregate.txt")
    X_aggregate_read = [[c[0] for c in x] for x in aggregate_read]
    Y_aggregate_read = [[c[1] for c in x] for x in aggregate_read]

    y_weights = y_answers
    for i in range(0, len(Y_aggregate_read)):
        for j in range(0, len(Y_aggregate_read[i])):
            y_weights[i][j].append(Y_aggregate_read[i][j])

    print(len(y_weights[0][0]))
    priority_array_input = outcome(X_train, y_weights)
    entity_value = np.sum(priority_array_input)/len(priority_array_input)
    priority_array_input_new = []
    for i in range(0,8):
        priority_array_input_new.append(entity_value)
    priority_array_input_new.append(1)
    priority_array_input = priority_array_input_new
    print(priority_array_input)
    priority_array_file.write(str(priority_array_input))
    priority_array_file.write("\n")
    ystarcap_label_encoding = []
    for i in range(0, len(ystarcap)):
        ystarcap_test_label_i = []
        for j in range(0, len(ystarcap[i])):
            label_ystarcap_one_hot = onehot_vector_encoding(ystarcap[i][j])
            ystarcap_test_label_i.append(label_ystarcap_one_hot)
        ystarcap_label_encoding.append(ystarcap_test_label_i)
    print(ystarcap_label_encoding[0])

    actual_label_one_hot = []
    for i in range(len(labels)):
        actual_label_one_hot_i = onehot_vector_encoding(labels[i])
        actual_label_one_hot.append(actual_label_one_hot_i)
    print(actual_label_one_hot)

    ystar_label_aggregate = []
    for i in range(0, len(ystar_test_label_encoding)):
        ystar_label_aggregate_i = []
        for j in range(0, len(ystar_test_label_encoding[i])):
            label_ystar_aggregate = ystar_label(ystar_test_label_encoding[i][j], ystarcap_label_encoding[i][j])
            ystar_label_aggregate_i.append(label_ystar_aggregate)
        ystar_label_aggregate.append(ystar_label_aggregate_i)
    print(ystar_label_aggregate[0])

    if (iteration > 0):
        distance_workers = []
        for i in range(0, worker_annotators_count):
            sentences_by_worker = sentences_worked_by_worker(i)
            if len(sentences_by_worker) != 0:
                distance_workers.append(no_of_annotations(i)/ sum_of_distance(i))
            else:
                distance_workers.append(1000)
        max_of_worker_distances = np.max(distance_workers) + 1e-7
        for i in range(0, worker_annotators_count):
            worker_weights[i] = -math.log(distance_workers[i] / max_of_worker_distances)

    cnn_x_answers = read_conll(DATA_PATH + 'cnn_predict_output_formatted.txt')
    cnn_x_words = [[c[0] for c in x] for x in cnn_x_answers]
    cnn_x_labels = [[c[1] for c in y] for y in cnn_x_answers]
    print(cnn_x_labels[0][0])

    # Data for model iteration
    cnn_test_ystar_label_aggregate = []
    cnn_test_ystarcap_label_encoding = []
    cnn_confidence_measurement = []
    number_of_annotations_cnn = []
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        cnn_confidence_measurement.append(confidence_measurement_data[sentence_number])
        number_of_annotations_cnn.append(int(number_of_annotations_sentence[sentence_number][0]))
        cnn_test_ystar_label_encoding_internal = []
        cnn_test_ystarcap_label_encoding_internal = []
        for j in range(0, len(ystar_label_aggregate[sentence_number])):
            cnn_test_ystar_label_encoding_internal.append(ystar_label_aggregate[sentence_number][j])
            cnn_test_ystarcap_label_encoding_internal.append(ystarcap_label_encoding[sentence_number][j])
        cnn_test_ystar_label_aggregate.append(cnn_test_ystar_label_encoding_internal)
        cnn_test_ystarcap_label_encoding.append(cnn_test_ystarcap_label_encoding_internal)
    print(cnn_test_ystar_label_aggregate[0])
    print(cnn_test_ystarcap_label_encoding[0])
    print(number_of_annotations_cnn[0])

    #commented
    if (iteration > 0):
        weight_of_worker_cnn = np.max(worker_weights)
        worker_weight_cnn[0] = weight_of_worker_cnn
        print(weight_of_worker_cnn)

    # Finding aggregate labels for all the sentences
    Label_sentence_agg = []
    for i in range(0, len(ystar_x_words)):
        Label_sentence = viterbi_sequence_labels(i)
        Label_sentence_agg.append(Label_sentence)
    print(Label_sentence_agg)

    # Updated Worker Weight
    # Creating data for next iteration

    updated_worker_weight = open("updated_worker_weight.txt", "a")

    for i in range(0, worker_annotators_count):
        worker_weight = worker_weights[i]
        updated_worker_weight.write(str(worker_weight))
        updated_worker_weight.write("\n")
    updated_worker_weight.close()


    weight_vector_this_iteration = np.asarray(worker_weights)

    # Updated Worker Weight
    # Creating data for next iteration
    updated_worker_weight_cnn = open("updated_rtheta.txt", "a")
    worker_weight_cnn = worker_weight_cnn[0]
    updated_worker_weight_cnn.write(str(worker_weight_cnn))
    updated_worker_weight_cnn.close()

    # Data for model iteration
    # Creating data for train
    total_sentences_aggregate = open("total_sentences_aggregate_cnn.txt", "a")
    for i in range(0, len(Label_sentence_agg)):
        for j in range(0, len(Label_sentence_agg[i])):
            total_sentences_aggregate.write(str(ystar_x_words[i][j]))
            total_sentences_aggregate.write(" ")
            total_sentences_aggregate.write(str(Label_sentence_agg[i][j]))
            total_sentences_aggregate.write("\n")
        total_sentences_aggregate.write("\n")
    total_sentences_aggregate.close()

    # Data for model iteration
    # Creating data for train
    train_sentences_aggregate = open("train_sentences_aggregate_cnn.txt", "a")
    for i in range(0, len(train_sentences)):
        sentence_number = train_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            train_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            train_sentences_aggregate.write(" ")
            train_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            train_sentences_aggregate.write("\n")
        train_sentences_aggregate.write("\n")
    train_sentences_aggregate.close()

    # Data for model iteration
    # Creating data for train
    test_sentences_aggregate = open("test_sentences_aggregate_cnn.txt", "a")
    for i in range(0, len(test_sentences)):
        sentence_number = test_sentences[i]
        for j in range(0, len(Label_sentence_agg[sentence_number])):
            test_sentences_aggregate.write(str(ystar_x_words[sentence_number][j]))
            test_sentences_aggregate.write(" ")
            test_sentences_aggregate.write(str(Label_sentence_agg[sentence_number][j]))
            test_sentences_aggregate.write("\n")
        test_sentences_aggregate.write("\n")
    test_sentences_aggregate.close()

    word_index = open("data/word_index.txt", "r")
    aggregated_file_updated = open("aggregated_file_updated.txt", "a")
    for line in word_index:
        if line not in ["\n", "\r\n"]:
            index = line.rstrip()
            index_numbers = index.split()
            sentence_number = int(index_numbers[0])
            word_number = int(index_numbers[1])
            aggregated_file_updated.write(str(ystar_x_words[sentence_number][word_number]))
            aggregated_file_updated.write(" ")
            aggregated_file_updated.write(str(Label_sentence_agg[sentence_number][word_number]))
            aggregated_file_updated.write("\n")
        if line in ["\n", "\r\n"]:
            aggregated_file_updated.write("\n")

    aggregated_file_updated.close()
    word_index.close()

    word_index = open("data/word_index.txt", "r")
    aggregated_file_updated = open("gt_file_updated.txt", "a")
    for line in word_index:
        if line not in ["\n", "\r\n"]:
            index = line.rstrip()
            index_numbers = index.split()
            sentence_number = int(index_numbers[0])
            word_number = int(index_numbers[1])
            aggregated_file_updated.write(str(ystar_x_words[sentence_number][word_number]))
            aggregated_file_updated.write(" ")
            aggregated_file_updated.write(str(gt_labels[sentence_number][word_number]))
            aggregated_file_updated.write("\n")
        if line in ["\n", "\r\n"]:
            aggregated_file_updated.write("\n")

    aggregated_file_updated.close()
    word_index.close()