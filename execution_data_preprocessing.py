#Required Imports are loaded in this section
import os
import shutil
import math
import numpy as np

#Functions are defined here

def entropy_calc(weight_of_worker, first_counter, second_counter):
    test = first_counter/second_counter
    entropy_test = weight_of_worker * test * math.log(test,2)
    return entropy_test
def entropy_calc_sentence(first_counter, second_counter):
    test = first_counter/second_counter
    entropy_test = test * math.log(test,2)
    return entropy_test

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

#In this section the inputs needed by the script are provided

#Provide the working directory, this is the location where this script is located.
CURRENT_WORKING_DIRECTORY = os.getcwd()
EXECUTION_DIRECTORY = CURRENT_WORKING_DIRECTORY + "/execution/"
ITERATION_DIRECTORY = EXECUTION_DIRECTORY + "/iteration0"

#Initial worker_weights
worker_weights_init = 0.7


#Execution begins from here

#Change to the working directory and create necessary folder structures
os.chdir(CURRENT_WORKING_DIRECTORY)
os.system('mkdir -p ' + EXECUTION_DIRECTORY)
os.system('mkdir -p ' + ITERATION_DIRECTORY)
os.chdir(ITERATION_DIRECTORY)
path_new = os.getcwd()
print(path_new)

#Create necessary folder structure for pre-processing
os.system('mkdir -p data')
os.system('mkdir -p dictionary_directory/dictionary_sentence_list')
os.system('mkdir -p dictionary_directory/dictionary_mv_sentence_list')
os.system('mkdir -p dictionary_directory/dictionary_gt_sentence_list')
os.system('mkdir individual_files')
os.system('mkdir -p dictionary_directory/dictionary_worker_sentence')
os.system('mkdir -p dictionary_directory/dictionary_sentence_worker')

#Copy required files for processing
conllevalpl = shutil.copy("../../conlleval.pl", "conlleval.pl")
conllevalpy = shutil.copy("../../conlleval.py", "conlleval.py")
conllevalpyc = shutil.copy("../../conlleval.pyc", "conlleval.pyc")
answers_copy = shutil.copy("../../NER/processed_test_data/answers.txt", "data/answers.txt")
groundtruth_copy = shutil.copy("../../NER/processed_test_data/Ground_Truth.txt", "data/Ground_Truth.txt")
mv_copy = shutil.copy("../../NER/processed_test_data/mv.txt", "data/mv.txt")


#Processing answers.txt

answers = open("data/answers.txt", "r")
counter = 1
for line in answers:
  if line not in ['\n', '\r\n']:
    sentence_count = open("dictionary_directory/dictionary_sentence_list/sentence_"+ str(counter) +".txt", "a")
    sentence_count.write(line)
    sentence_count.close()
  if line in ['\n', '\r\n']:
    counter = counter+1
answers.close()

#Processing mv.txt

mv = open("data/mv.txt", "r")
counter = 1
for line in mv:
  if line not in ['\n', '\r\n']:
    sentence_count = open("dictionary_directory/dictionary_mv_sentence_list/sentence_"+ str(counter) +".txt", "a")
    sentence_count.write(line)
    sentence_count.close()
  if line in ['\n', '\r\n']:
    counter = counter+1
mv.close()

#Processing Ground_Truth.txt

gt = open("data/Ground_Truth.txt", "r")
counter = 1
for line in gt:
  if line not in ['\n', '\r\n']:
    sentence_count = open("dictionary_directory/dictionary_gt_sentence_list/sentence_"+ str(counter) +".txt", "a")
    sentence_count.write(line)
    sentence_count.close()
  if line in ['\n', '\r\n']:
    counter = counter+1
gt.close()

#Get sentence and worker count from the dataset

no_of_sentences = len(os.listdir("dictionary_directory/dictionary_sentence_list"))
print("Number of sentences " + str(no_of_sentences))
test_read = read_conll("data/answers.txt")
X_test = [[c[0] for c in x] for x in test_read]
y_test = [[c[1:] for c in y] for y in test_read]
no_of_workers = len(y_test[0][0])
print("Number of workers " + str(no_of_workers))

#Finding relationship between sentences and workers
#Creating sentence list file for easy access
sentence_list = open("individual_files/sentence_all_list.txt", "a")
sentence_id = 1
for id in range(1,no_of_sentences+1,1):
    a = "sentence_"
    b = ".txt"
    sentence_file = a + str(sentence_id) + b
    sentence_list.write(sentence_file + "\n")
    sentence_id = sentence_id + 1
sentence_list.close()

#Creating worker list file for easy access
worker_list = open("individual_files/worker_all_list.txt", "a")
worker_id = 1
for id in range(1,no_of_workers+1,1):
    a = "Worker_"
    worker_file = a + str(worker_id)
    worker_list.write(worker_file + "\n")
    worker_id = worker_id + 1
worker_list.close()

#Creating word mapping file
worker_word_mapping = open("individual_files/worker_word_mapping.txt", "a")
worker = [0]*no_of_workers

for sentence_id in range(1,no_of_sentences+1,1):
    counter = 0
    a = "sentence_"
    b = ".txt"
    sentence_file = a + str(sentence_id) + b
    sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
    for line1 in sentence_loop:
        words = line1.split()
        for id in range(1,no_of_workers+1,1):
            if str(words[id]) != '?':
                worker[id-1] = worker[id-1] + 1
    sentence_loop.close()
c = "worker_"
for worker_id in range(1,no_of_workers+1,1):
    worker_word_mapping.write( c + str(worker_id) + "\t" + str(worker[worker_id-1]) + "\n")

worker_word_mapping.close()

#Creating worker sentence mapping
worker_sentence_mapping = open("individual_files/worker_sentence_mapping.txt", "a")
worker = [0]*no_of_workers

for sentence_id in range(1,no_of_sentences+1,1):
    counter = 0
    a = "sentence_"
    b = ".txt"
    sentence_file = a + str(sentence_id) + b
    sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
    for line1 in sentence_loop:
        words = line1.split()
    for id in range(1,no_of_workers+1,1):
        if str(words[id]) != '?':
            worker[id-1] = worker[id-1] + 1
    sentence_loop.close()
c = "worker_"
for worker_id in range(1,no_of_workers+1,1):
    worker_sentence_mapping.write( c + str(worker_id) + "\t" + str(worker[worker_id-1]) + "\n")

worker_sentence_mapping.close()

#Creating worker sentence directory
for i in range(1,no_of_workers+1,1):
    counter = 0
    worker_file = open('dictionary_directory/dictionary_worker_sentence/worker_'+str(i) +'.txt', "a")
    for j in range(1,no_of_sentences+1,1):
        a = "sentence_"
        b = ".txt"
        sentence_file = a + str(sentence_id) + b
        sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
        for line1 in sentence_loop:
            words = line1.split()
            if str(words[i]) != '?':
                counter = counter + 1
        if counter > 0:
            worker_file.write(str(sentence_file) + "\n")
        sentence_loop.close()
    worker_file.close()

#Creating sentence worker mapping

sentence_worker_mapping = open("individual_files/sentence_worker_mapping.txt", "a")

for sentence_id in range(1,no_of_sentences+1,1):
    a = "sentence_"
    b = ".txt"
    sentence_file = a + str(sentence_id) + b
    sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
    for line1 in sentence_loop:
        counter = 0
        words = line1.split()
        for word in words:
            if word == 'O' or word == 'B-LOC' or word == 'B-MISC' or word == 'I-MISC' or word == 'B-ORG' or word == 'B-PER' or word == 'I-PER' or word == 'I-LOC' or word == 'I-ORG':
                counter = counter + 1
    sentence_loop.close()
    sentence_worker_mapping.write( str(sentence_file) + "\t" + str(counter) + "\n")

sentence_worker_mapping.close()

#Creating sentence worker dictionary
for sentence_id in range(1,no_of_sentences+1,1):
    worker = [0]*no_of_workers
    a = "sentence_"
    b = ".txt"
    sentence_file = a + str(sentence_id) + b
    sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
    for line1 in sentence_loop:
        words = line1.split()
    for id in range(1, no_of_workers+1, 1):
        if str(words[id]) != '?':
            worker[id - 1] = worker[id - 1] + 1
    sentence_write = open("dictionary_directory/dictionary_sentence_worker/" + str(sentence_file) + "", "a")
    c = "worker_"
    for id in range(1, no_of_workers+1, 1):
        if worker[id - 1] > 0:
            sentence_write.write(c + str(id) + "\n")
    sentence_write.close()
    sentence_loop.close()

#Creating sentence word mapping
sentences_word_mapping = open("individual_files/sentences_word_mapping.txt", "a")
for sentence_id in range(1,no_of_sentences+1,1):
    counter = 0
    a = "sentence_"
    b = ".txt"
    sentence_file = a + str(sentence_id) + b
    sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
    for line1 in sentence_loop:
        if line1 not in ['\n', '\r\n']:
            counter = counter + 1
    sentences_word_mapping.write(str(sentence_file) + "\t" + str(counter) + "\n")
    sentence_loop.close()
sentences_word_mapping.close()

#calculating the count of words to be considered for each sentence
sentence_list = open("individual_files/sentence_all_list.txt", "r")
number_of_annotations_file = open("individual_files/number_of_annotations.txt", "a")
word_index = open("individual_files/word_index.txt", "a")
sentence_count = 0
for line in sentence_list:
    word_count = 0
    entropy = 0
    weight_of_worker = 1
    number_of_annotations = 0
    total_counter = 0
    end_counter = 0
    end_counter1 = 0
    end_counter2 = 0
    end_counter3 = 0
    end_counter4 = 0
    end_counter5 = 0
    end_counter6 = 0
    end_counter7 = 0
    end_counter8 = 0
    sentence_file = line.rstrip()
    sentence_loop = open("dictionary_directory/dictionary_sentence_list/" + str(sentence_file) + "", "r")
    for line1 in sentence_loop:
        entropy_word_wise = 0
        counter = 0
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        counter5 = 0
        counter6 = 0
        counter7 = 0
        counter8 = 0
        counter9 = 0
        words = line1.split()
        for word in words:
          if word == 'O' or word == 'B-LOC' or word == 'B-MISC' or word == 'I-MISC' or word == 'B-ORG' or word == 'B-PER' or word == 'I-PER' or word == 'I-LOC' or word == 'I-ORG':
            counter = counter + 1
        for word in words:
          if word == 'O':
            counter1 = counter1 + 1
          if word == 'B-LOC':
            counter2 = counter2 + 1
          if word == 'B-MISC':
            counter3 = counter3 + 1
          if word == 'I-MISC':
            counter4 = counter4 + 1
          if word == 'B-ORG':
            counter5 = counter5 + 1
          if word == 'B-PER':
            counter6 = counter6 + 1
          if word == 'I-PER':
            counter7 = counter7 + 1
          if word == 'I-LOC':
            counter8 = counter8 + 1
          if word == 'I-ORG':
            counter9 = counter9 + 1
        if counter > 0 and counter1 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter1, counter)
        if counter > 0 and counter2 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter2, counter)
        if counter > 0 and counter3 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter3, counter)
        if counter > 0 and counter4 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter4, counter)
        if counter > 0 and counter5 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter5, counter)
        if counter > 0 and counter6 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter6, counter)
        if counter > 0 and counter7 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter7, counter)
        if counter > 0 and counter8 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter8, counter)
        if counter > 0 and counter9 > 0:
            entropy_word_wise = entropy_word_wise - entropy_calc(weight_of_worker, counter9, counter)
        if counter > 0:
            total_counter = total_counter + (counter)
        if entropy_word_wise == 0 and counter1 == 0:
            number_of_annotations = number_of_annotations + 1
            word_index.write(str(sentence_count) + " " + str(word_count) + "\n")
        elif entropy_word_wise > 0:
            number_of_annotations = number_of_annotations + 1
            word_index.write(str(sentence_count) + " " + str(word_count) + "\n")
        word_count = word_count + 1
    number_of_annotations_file.write(str(line.rstrip())+ " " + str(number_of_annotations) + "\n")
    number_of_annotations_file.write("\n")
    word_index.write("\n")
    sentence_loop.close()
    sentence_count = sentence_count + 1
number_of_annotations_file.close()
sentence_list.close()
word_index.close()

#Creating initial worker_weights and cnn_weight_file

worker_weights = [worker_weights_init]*no_of_workers
worker_weight_write_file = open("data/worker_weight.txt", "w")
for i in range(0,len(worker_weights)):
    worker_weight_write_file.write(str(worker_weights[i]))
    worker_weight_write_file.write("\n")
worker_weight_write_file.close()

cnn_worker_weights = worker_weights_init
cnn_worker_weight_write_file = open("data/r_theta.txt", "w")
cnn_worker_weight_write_file.write(str(cnn_worker_weights))
cnn_worker_weight_write_file.close()