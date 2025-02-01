import os
import argparse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from eval_phish import BertClassifier
from tqdm import tqdm

torch.serialization.add_safe_globals([BertClassifier])

def main():
    parser = argparse.ArgumentParser(description="Generate address embeddings from a pretrained BERT model.")
    parser.add_argument("--tsv_file", type=str, help="Path to the input TSV file (must contain two columns: address, text).")
    parser.add_argument("--pretrained_model_path", type=str, help="Directory or name of the pretrained BERT model.")
    parser.add_argument("--model_path", type=str, help="Path to the fine-tuned model.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the output .npy files.")
    parser.add_argument("--max_length", type=int, help="BERT tokenizer truncation length.")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    addresses = []
    texts = []
    with open(args.tsv_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            cols = line.strip().split('\t')
            if len(cols) < 2:
                continue
            addr = cols[0]
            text = cols[1]
            addresses.append(addr)
            texts.append(text)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    model = BertClassifier(pretrained_model_path=args.pretrained_model_path)
    model.eval()
    model.to("cuda:3" if torch.cuda.is_available() else "cpu")

    all_embeddings = []

    def get_batch(start, end):
        batch_texts = texts[start:end]
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors='pt'
        )
        return encoding

    device = next(model.parameters()).device

    num_samples = len(texts)
    batch_size = args.batch_size
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="generating embeddings"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_encoding = get_batch(start_idx, end_idx)

            input_ids = batch_encoding["input_ids"].to(device)
            attention_mask = batch_encoding["attention_mask"].to(device)

            _, batch_embeddings = model(input_ids, attention_mask=attention_mask)
            batch_embeddings = batch_embeddings.cpu().numpy()

            all_embeddings.append(batch_embeddings)

    all_embeddings = np.concatenate(all_embeddings, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)

    addr_npy_path = os.path.join(args.output_dir, "address.npy")
    emb_npy_path = os.path.join(args.output_dir, "embedding.npy")

    np.save(addr_npy_path, np.array(addresses, dtype=object))
    np.save(emb_npy_path, all_embeddings)

if __name__ == "__main__":
    main()