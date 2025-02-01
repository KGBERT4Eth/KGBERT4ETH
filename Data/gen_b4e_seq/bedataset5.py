import pickle as pkl
import tqdm

with open("bert4eth_trans4.pkl", "rb") as f:
    trans_seq = pkl.load(f)

# Read addresses from txt file
with open("phisher_account.txt", "r") as f:
    addresses_in_file = {line.strip() for line in f}

next_trans_seq = {}
# Iterate through each address and its transaction sequence
for address, transactions in tqdm.tqdm(trans_seq.items(), desc="Processing transaction data"):
    prefix = '1' if address in addresses_in_file else '0'

    trans_seq_str = prefix
    for transaction in transactions:
        # Format transaction information and add to string
        trans_str = f" amount: {transaction[0]} in_out: {transaction[1]} 2-gram: {transaction[2]} 3-gram: {transaction[3]} 4-gram: {transaction[4]} 5-gram: {transaction[5] } "
        trans_seq_str += trans_str
    # Add a period at the end of the string
    trans_seq_str = trans_seq_str.strip() + '.'
    # Update next_trans_seq dictionary
    next_trans_seq[address] = [trans_seq_str]

# Save the modified trans_seq to file
with open("bert4eth_trans5.pkl", "wb") as f:
    pkl.dump(next_trans_seq, f)

print(1)