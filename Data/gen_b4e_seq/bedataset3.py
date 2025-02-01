import pickle as pkl
import tqdm

with open("bert4eth_trans2.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in trans_seq.items():
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions

for address, transactions in tqdm.tqdm(trans_seq.items(), desc="处理n-gram数据"):
    for i in range(len(transactions)):
        # Initialize a list to hold the n-gram time differences for this transaction
        n_gram_times = []
        for n in range(2, 6):  # 处理2-gram到5-gram
            if i >= n - 1:
                # 计算n-gram时间差并添加到n_gram_times列表
                n_gram_time_diff = transactions[i][1]-transactions[i - n + 1][1]
                n_gram_times.append(n_gram_time_diff)
            else:
                # 对于序列中前n-1个交易，无法计算n-gram时间差，添加0
                n_gram_times.append(0)
        # 将n-gram时间差直接添加到当前交易的末尾
        transactions[i].extend(n_gram_times)

with open("bert4eth_trans3.pkl", "wb") as f:
    pkl.dump(trans_seq, f)

print(1)
