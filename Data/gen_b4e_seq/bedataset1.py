import random
import pickle as pkl
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pkl.dump(data, file)


# cai yang n=2*p
# 假设a是已经存在的字典
with open("trans_seq.pkl", "rb") as f:
    a = pkl.load(f)
# 初始化b字典
b = {}

# 步骤1: 读取欺诈账户地址
with open("phisher_account.txt", "r") as f:
    fraud_accounts = [line.strip() for line in f]

# 步骤2: 从a中提取欺诈账户的交易信息到b中
for address in fraud_accounts:
    if address in a:
        b[address] = a[address]

# 步骤3: 从a中提取两倍于欺诈账户数量的正常账户信息到b中
normal_accounts = [address for address in a if address not in fraud_accounts]
selected_normal_accounts = random.sample(normal_accounts, len(b))

for address in selected_normal_accounts:
    b[address] = a[address]

save_data(b, 'bert4eth_trans1.pkl')
print(1)
# 现在字典b包含了所有欺诈账户的交易信息和两倍数量的正常账户交易信息
