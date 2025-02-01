import random
import pickle as pkl

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pkl.dump(data, file)

with open("trans_seq.pkl", "rb") as f:
    a = pkl.load(f)

b = {}

with open("phisher_account.txt", "r") as f:
    fraud_accounts = [line.strip() for line in f]

for address in fraud_accounts:
    if address in a:
        b[address] = a[address]

normal_accounts = [address for address in a if address not in fraud_accounts]
selected_normal_accounts = random.sample(normal_accounts, len(b))

for address in selected_normal_accounts:
    b[address] = a[address]

save_data(b, 'bert4eth_trans1.pkl')
print(1)