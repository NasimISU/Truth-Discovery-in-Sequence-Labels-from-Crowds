from sklearn.utils import class_weight
import os
import numpy as np

CURRENT_WORKING_DIRECTORY = os.getcwd()
DATA_PATH = CURRENT_WORKING_DIRECTORY + "/execution/iteration0/data/"

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

def get_tokens_for_labels(x, y, entity_label, non_entity_label):
    entity_label = entity_label
    non_entity_label = non_entity_label
    tokens_for_labels = open(DATA_PATH + "tokens_for_labels_" + entity_label + ".txt", "w")
    X_tokens = []
    Y_tokens = []
    for i in range(0,len(y)):
        for j in range(0,len(y[i])):
            count_non_entity = 0
            count_entity = 0
            total = 0
            count_question = 0
            for k in range(0,len(y[i][j])):
                if y[i][j][k] == non_entity_label:
                    count_non_entity = count_non_entity + 1
                elif y[i][j][k] == entity_label:
                    count_entity = count_entity + 1
                elif y[i][j][k] == '?':
                    count_question = count_question + 1
            total = count_non_entity + count_entity + count_question
            if total == len(y[i][j]) and count_entity!=0 and count_non_entity!=0:
                X_tokens.append(x[i][j])
                tokens_for_labels.write(x[i][j])
                Y_tokens.append(y[i][j])
                for k in range(0,len(y[i][j])):
                    tokens_for_labels.write(" ")
                    tokens_for_labels.write(y[i][j][k])
                tokens_for_labels.write("\n")
    return X_tokens, Y_tokens


answers = read_conll(DATA_PATH + "answers.txt")
X_answers = [[c[0] for c in x] for x in answers]
Y_answers = [[c[1:] for c in x] for x in answers]

mv = read_conll(DATA_PATH + "mv.txt")
X_mv = [[c[0] for c in x] for x in mv]
Y_mv = [[c[1] for c in x] for x in mv]

for i in range(0,len(Y_mv)):
    for j in range(0,len(Y_mv[i])):
        Y_answers[i][j].append(Y_mv[i][j])

def compute_class_weight(X_token, Y_token):
    computed_class_weight = []
    output_class_weight = 0
    for i in range(0,len(X_token)):
        y = []
        for j in range(0,len(Y_token[i])):
            if Y_token[i][j] == 'O':
                y.append(0)
            elif Y_token[i][j] == '?':
                continue
            else:
                y.append(1)
        print(y)
        y.sort()
        print(y)
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
        class_weights = class_weights*(1/class_weights[0])
        computed_class_weight.append(class_weights[1])
    output_class_weight = (np.sum(computed_class_weight))/len(computed_class_weight)
    return output_class_weight

def outcome(X_answers, Y_answers):
    final_class_weights = []
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'B-LOC', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'B-PER', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'B-ORG', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'B-MISC', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'I-LOC', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'I-PER', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'I-ORG', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    X_tokens, Y_tokens = get_tokens_for_labels(X_answers, Y_answers, 'I-MISC', 'O')
    weight = compute_class_weight(X_tokens, Y_tokens)
    final_class_weights.append(weight)
    final_class_weights.append(1)
    return final_class_weights