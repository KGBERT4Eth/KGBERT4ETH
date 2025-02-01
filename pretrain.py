import torch
import datetime
import os
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import BertTokenizer
from model.KGETHBERT_Model import KGETHBERT
from model.Dataset import JointDataset, Bm25BertDataset, KGLP_Dataset
from utils import load_pkl

account_texts = load_pkl('dev_corpus.pkl')
doc_token_bm25_scores = load_pkl('list_token_bm25_scores.pkl')
kg_triples = load_pkl('triples.pkl')
account_triples_idx = load_pkl('account_triples_indices.pkl')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

relation_set = set()
entity_set = set()

for (h, r, t) in kg_triples:
    relation_set.add(r)
    entity_set.add(h)
    entity_set.add(t)

relation2id = {rel: idx for idx, rel in enumerate(sorted(relation_set))}

num_entities = len(entity_set)
num_relations = len(relation2id)

kg_triples_id = [(h, relation2id[r], t) for (h, r, t) in kg_triples]

mlm_dataset = Bm25BertDataset(
    texts=account_texts,
    doc_token_bm25_scores=doc_token_bm25_scores,
    tokenizer=tokenizer,
    bm25_top_p=0.3,  # top30%
    max_length=64
)

kg_dataset = KGLP_Dataset(triples=kg_triples_id, num_entities=num_entities)

joint_dataset = JointDataset(mlm_dataset, kg_dataset, account_triples_idx)
joint_dataloader = DataLoader(joint_dataset, batch_size=256, shuffle=False)

model = KGETHBERT(
    bert_model_name="bert-base-uncased",
    num_entities=num_entities,
    num_relations=num_relations,
    emb_dim=32
)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-5)

current_time = datetime.datetime.now().strftime("%m_%d_%H_%M")
save_dir = f"Train_output/{current_time}"
os.makedirs(save_dir, exist_ok=True)

EPOCHS = 2
for epoch in range(EPOCHS):
    model.train()
    total_loss_val = 0.0
    total_mlm_loss_val = 0.0
    total_kg_loss_val = 0.0

    for batch in tqdm(joint_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        triple_pos = torch.stack(batch["triple_pos"], dim=1).to(device)
        triple_neg = torch.stack(batch["triple_neg"], dim=1).to(device)

        optimizer.zero_grad()
        total_loss, mlm_loss, kg_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            triple_pos=triple_pos,
            triple_neg=triple_neg
        )

        total_loss.backward()
        optimizer.step()

        total_loss_val += total_loss.item()
        total_mlm_loss_val += mlm_loss.item()
        total_kg_loss_val += kg_loss.item()

    avg_loss = total_loss_val / len(joint_dataloader)
    avg_mlm_loss = total_mlm_loss_val / len(joint_dataloader)
    avg_kg_loss = total_kg_loss_val / len(joint_dataloader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, total_loss={avg_loss:.4f}, mlm_loss={avg_mlm_loss:.4f}, kg_loss={avg_kg_loss:.4f}"
    )

    # if (epoch + 1) % 5 == 0:
    #     model.save_pretrained(save_dir)

model.save_pretrained(save_dir)

print("end")

