from tqdm import tqdm
import pickle as pkl
from scipy.sparse import csr_matrix

with open("bert4eth_trans1.pkl", "rb") as f:
    trans_seq = pkl.load(f)

address_to_idx = {}
idx_to_address = {}
undirect_trans_freq = {}
index = 0

for address, transactions in tqdm(trans_seq.items()):
    modified_transactions = []
    for transaction in transactions:
        # 从每个交易中移除索引为1和5的元素
        modified_transaction = [transaction[i] for i in range(len(transaction)) if i not in [1, 5]]
        modified_transactions.append(modified_transaction)
    trans_seq[address] = modified_transactions  # 更新地址的交易列表为修改后的交易列表

    for transaction in modified_transactions:
        vice_address = transaction[0]  # 更新后的transaction[1]现在是transaction[0]
        timestamp = transaction[1]  # 更新后的transaction[2]现在是transaction[1]
        amount = transaction[2]  # 更新后的transaction[3]现在是transaction[2]
        io_flag = transaction[3]  # 更新后的transaction[4]现在是transaction[3]
        if address not in address_to_idx:
            address_to_idx[address] = index
            idx_to_address[index] = address
            index += 1

        if vice_address not in address_to_idx:
            address_to_idx[vice_address] = index
            idx_to_address[index] = vice_address
            index += 1
        pair = tuple(sorted([address_to_idx[address], address_to_idx[vice_address]]))
        if pair in undirect_trans_freq:
            undirect_trans_freq[pair] += 1
        else:
            undirect_trans_freq[pair] = 1

num_of_address = len(address_to_idx)
data, row_indices, col_indices = [], [], []
for (addr1, addr2), freq in tqdm(undirect_trans_freq.items()):
    row_indices.append(addr1)
    col_indices.append(addr2)
    data.append(freq)
adj = csr_matrix((data, (row_indices, col_indices)), shape=(num_of_address, num_of_address))

with open('adj.pkl', 'wb') as f:
    pkl.dump(adj, f)

with open('address_to_idx.pkl', 'wb') as f:
    pkl.dump(address_to_idx, f)

with open('idx_to_address.pkl', 'wb') as f:
    pkl.dump(idx_to_address, f)

with open('bert4eth_trans2.pkl', 'wb') as f:
    pkl.dump(trans_seq, f)
print(1)