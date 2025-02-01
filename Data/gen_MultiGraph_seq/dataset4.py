import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def add_n_grams(accounts):
    for address, transactions in tqdm.tqdm(accounts.items()):
        for n in range(2, 6):
            gram_key = f"{n}-gram"
            for i in range(len(transactions)):
                if i < n-1:
                    transactions[i][gram_key] = 0
                else:
                    transactions[i][gram_key] = transactions[i]['timestamp'] - transactions[i-n+1]['timestamp']

accounts_data = load_data('transactions3.pkl')

add_n_grams(accounts_data)

save_data(accounts_data, 'transactions4.pkl')

