import pickle
from sklearn.model_selection import train_test_split

# Load data from bert4eth_trans5.pkl file
with open('bert4eth_trans5.pkl', 'rb') as file:
    transactions_dict = pickle.load(file)

with open('address_to_idx.pkl', 'rb') as file:
    address_to_idx = pickle.load(file)

# Convert dictionary to list, each element is a "tag sentence" formatted string
transactions = []
for key, value_list in transactions_dict.items():
    for value in value_list:
        index = address_to_idx[key]
        transactions.append(f"{index} {value}")

# Define data split ratios
train_size = 0.8
validation_size = 0.1
test_size = 0.1

# Split data into training set and remaining part
train_data, temp_data = train_test_split(transactions, train_size=train_size, random_state=42)

# Split remaining part into validation set and test set
validation_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + validation_size), random_state=42)

# Function to save training and validation data to TSV file
def save_to_tsv_train_dev(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tlabel\tsentence\n")
        for line in data:
            index, rest = line.split(' ', 1)
            tag, sentence = rest.split(' ', 1)
            file.write(f"{index}\t{tag}\t{sentence}\n")

# Function to save test data to TSV file
def save_to_tsv_test(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tlabel\tsentence\n")
        for idx, line in enumerate(data):
            index, rest = line.split(' ', 1)
            tag, sentence = rest.split(' ', 1)
            file.write(f"{index}\t{tag}\t{sentence}\n")

# Save training, validation, and test sets
save_to_tsv_train_dev(train_data, 'train.tsv')
save_to_tsv_train_dev(validation_data, 'dev.tsv')
save_to_tsv_test(test_data, 'test.tsv')

print("Files saved: train.tsv, dev.tsv, test.tsv")