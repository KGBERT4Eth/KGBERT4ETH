import pickle as pkl
import tqdm

with open("bert4eth_trans4.pkl", "rb") as f:
    trans_seq = pkl.load(f)

# 读取txt文件中的地址
with open("phisher_account.txt", "r") as f:
    addresses_in_file = {line.strip() for line in f}

next_trans_seq = {}
# 遍历每个地址及其交易序列
for address, transactions in tqdm.tqdm(trans_seq.items(), desc="处理交易数据"):
    prefix = '1' if address in addresses_in_file else '0'

    trans_seq_str = prefix
    for transaction in transactions:
        # 格式化交易信息并添加到字符串中
        trans_str = f" amount: {transaction[0]} in_out: {transaction[1]} 2-gram: {transaction[2]} 3-gram: {transaction[3]} 4-gram: {transaction[4]} 5-gram: {transaction[5] } "
        trans_seq_str += trans_str
    # 在字符串末尾添加一个句号
    trans_seq_str = trans_seq_str.strip() + '.'
    # 更新trans_seq字典
    next_trans_seq[address] = [trans_seq_str]

# 保存修改后的trans_seq到文件
with open("bert4eth_trans5.pkl", "wb") as f:
    pkl.dump(next_trans_seq, f)

print(1)