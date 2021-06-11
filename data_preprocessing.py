import os

# Data Directories
CURRENT_WORKING_DIRECTORY = os.getcwd()
DATA_DIRECTORY = CURRENT_WORKING_DIRECTORY + "/NER/"
WRITE_DIRECTORY = DATA_DIRECTORY + "/original_data"
DATA_PATH =  CURRENT_WORKING_DIRECTORY + "/crf-ma-datasets/"

WORKER_DATA_PATH = DATA_PATH + "worker_data/"
SENTENCE_DATA_PATH = DATA_PATH + "worker_sentence_data/"
EXTRACTED_DATA = DATA_PATH + "/mturk/extracted_data/"
GROUND_TRUTH_DATA = DATA_PATH + "/mturk/ground_truth/"
DREDZE_DATA = DATA_PATH + "/mturk/dredze_format/"

# Functions
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

def array_to_sentence(array):
    sentences = ' '.join(array)
    return sentences

#Create Data Directories

if not os.path.exists(DATA_DIRECTORY):
    os.mkdir(DATA_DIRECTORY)
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)
if not os.path.exists(SENTENCE_DATA_PATH):
    os.mkdir(SENTENCE_DATA_PATH)
if not os.path.exists(WORKER_DATA_PATH):
    os.mkdir(WORKER_DATA_PATH)
if not os.path.exists(WRITE_DIRECTORY):
    os.mkdir(WRITE_DIRECTORY)

# Get all the files
files = os.listdir(EXTRACTED_DATA)

#Change the directory
os.chdir(EXTRACTED_DATA)

for file in files:
    if file != ".DS_Store":
        worker_file = EXTRACTED_DATA + file
        worker_files = os.listdir(worker_file)
        os.chdir(worker_file)
        data_file_path = WORKER_DATA_PATH + file + ".txt"
        data_file = open(data_file_path, "w")
        for txt_file in worker_files:
            read_file = open(txt_file, "r", encoding='windows-1252')
            for line in read_file:
                data_file.write(line)
            read_file.close()
        data_file.close()

#
files = os.listdir(GROUND_TRUTH_DATA)
combined_file = open(WRITE_DIRECTORY + "/ground_truth.txt", "w")
os.chdir(GROUND_TRUTH_DATA)
print(os.listdir(GROUND_TRUTH_DATA))

for file in files:
    read_file = open(file, "r", encoding='windows-1252')
    for line in read_file:
        combined_file.write(line)
    read_file.close()

combined_file.close()

worker_files = os.listdir(WORKER_DATA_PATH)
for file in worker_files:
    file_name = WORKER_DATA_PATH + file
    worker_answers = read_conll(file_name)
    write_file = open(SENTENCE_DATA_PATH + file, "w")
    X_test = [[c[0] for c in x] for x in worker_answers]
    for i in range(0, len(X_test)):
        output = array_to_sentence(X_test[i])
        write_file.write(output)
        write_file.write("\n")
    write_file.close()
#
#
answers_file_path = WRITE_DIRECTORY + "/answers_sentence.txt"
sentence_files = os.listdir(SENTENCE_DATA_PATH)
answers_file_write = open(answers_file_path, "w")
answers_file_write.close()

for file in sentence_files:
    file_name = SENTENCE_DATA_PATH + file
    worker_sentence_file = open(file_name, "r")
    for line in worker_sentence_file:
        answers_file_read = open(answers_file_path, "r")
        sentence_list = answers_file_read.readlines()
        answers_file_read.close()
        found = False
        for sentence in sentence_list:
            if line in sentence:
                found = True
        if not found:
            answers_file_append = open(answers_file_path, "a")
            answers_file_append.write(line)
            answers_file_append.close()
    worker_sentence_file.close()

trainset = read_conll(DREDZE_DATA + "trainset_mturk.dredze.txt")
X_train = [[c[0] for c in x] for x in trainset]
print("Length of train set: " + str(len(X_train)))

answers_file_train = open(WRITE_DIRECTORY + "/answers_file_train.txt", "w")

for i in range(0,len(trainset)):
    for j in range(0,len(trainset[i])):
        for k in range(0,len(trainset[i][j])):
            if trainset[i][j][k] not in ['NO_POS', 'NO_CHUNK', '1']:
                answers_file_train.write(trainset[i][j][k])
                answers_file_train.write(" ")
        answers_file_train.write("\n")
    answers_file_train.write("\n")

answers_file_train.close()

testset = read_conll(DREDZE_DATA + "testset_mturk.dredze.txt")
X_test = [[c[0] for c in x] for x in testset]
print("Length of test set: " + str(len(X_test)))

gt_file_train = open(WRITE_DIRECTORY + "/gt_file_test.txt", "w")

for i in range(0,len(testset)):
    for j in range(0,len(testset[i])):
        for k in range(0,len(testset[i][j])):
            if testset[i][j][k] not in ['NO_POS', 'NO_CHUNK', '1']:
                gt_file_train.write(testset[i][j][k])
                gt_file_train.write(" ")
        gt_file_train.write("\n")
    gt_file_train.write("\n")

gt_file_train.close()

answers = read_conll(WRITE_DIRECTORY + "/answers_file_train.txt")
X_answers = [[c[0] for c in x] for x in answers]

ground_truth = read_conll(WRITE_DIRECTORY + "/ground_truth.txt")

X_gt = [[c[0] for c in x] for x in ground_truth]
Y_gt = [[c[1] for c in x] for x in ground_truth]

gt_file_train = open(WRITE_DIRECTORY + "/gt_file_train.txt", "w")

for i in range(0, len(X_answers)):
    answers_sentence = array_to_sentence(X_answers[i])
    for j in range(0,len(X_gt)):
        gt_sentence = array_to_sentence(X_gt[j])
        found = False
        if str(answers_sentence) == str(gt_sentence):
            found = True
        if found:
            if len(X_answers[i]) == len(X_gt[j]):
                for a in range(0,len(X_answers[i])):
                    gt_file_train.write(X_answers[i][a])
                    gt_file_train.write(" ")
                    gt_file_train.write(Y_gt[j][a])
                    gt_file_train.write("\n")
                gt_file_train.write("\n")


gt_file_train.close()

answers = read_conll(WRITE_DIRECTORY + "/answers_file_train.txt")
X_answers = [[c[0] for c in x] for x in answers]

train_gt = read_conll(WRITE_DIRECTORY + "/gt_file_train.txt")
X_train_gt = [[c[0] for c in x] for x in X_answers]

print("X_answers length: " + str(len(X_answers)))

print("X_train_gt length: " + str(len(X_train_gt)))
