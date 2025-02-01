import pickle as pkl
import tqdm

with open("bert4eth_trans2.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in trans_seq.items():
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions

for address, transactions in tqdm.tqdm(trans_seq.items(), desc="Processing n-gram data"):
    for i in range(len(transactions)):
        # Initialize a list to hold the n-gram time differences for this transaction
        n_gram_times = []
        for n in range(2, 6):  # Process 2-gram to 5-gram
            if i >= n - 1:
                # Calculate n-gram time difference and add to n_gram_times list
                n_gram_time_diff = transactions[i][1] - transactions[i - n + 1][1]
                n_gram_times.append(n_gram_time_diff)
            else:
                # For the first n-1 transactions in the sequence, cannot calculate n-gram time difference, add 0
                n_gram_times.append(0)
        # Add n-gram time differences directly to the end of the current transaction
        transactions[i].extend(n_gram_times)

with open("bert4eth_trans3.pkl", "wb") as f:
    pkl.dump(trans_seq, f)

print(1)