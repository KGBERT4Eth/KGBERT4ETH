import pickle
import random
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys()):
        random.shuffle(accounts[address])

# 加载数据
accounts_data = load_data('transactions5.pkl')

# 打乱交易数据
shuffle_transactions(accounts_data)

# 保存数据
save_data(accounts_data, 'transactions6.pkl')

