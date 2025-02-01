import pickle as pkl
import tqdm

with open("bert4eth_trans3.pkl", "rb") as f:
    trans_seq = pkl.load(f)
# 遍历每个地址的交易序列
for address, transactions in tqdm.tqdm(trans_seq.items(), desc="处理交易数据"):
    for i in range(len(transactions)):
        # 移除每条交易的第一个信息
        transactions[i] = transactions[i][2:]
        # 检查第四个信息（原本的第五个信息，因为已经移除了一个）
        if transactions[i][1] == 'IN':
            transactions[i][1] = 0
        elif transactions[i][1] == 'OUT':
            transactions[i][1] = 1

# 保存修改后的交易序列数据
with open("bert4eth_trans4.pkl", "wb") as f:
    pkl.dump(trans_seq, f)
print(1)