import argparse
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import BertTokenizer, BertModel
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


class DevDataset(Dataset):
    def __init__(self, tsv_file, tokenizer, max_len=64):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(tsv_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx == 0:
                    continue
                cols = line.strip().split('\t')
                if len(cols) < 3:
                    continue
                label = int(cols[1])
                text = cols[2]
                self.samples.append((label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_path, num_labels=6, freeze_bert=False):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits, pooled_output


def compute_metrics(labels, preds, probs):
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    try:
        num_classes = probs.shape[1]
        labels_one_hot = np.eye(num_classes)[labels]
        auc_roc = roc_auc_score(labels_one_hot, probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc_roc = 0.0

    cm = confusion_matrix(labels, preds)

    return {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC-ROC': auc_roc,
        'Confusion_Matrix': cm.tolist()
    }


def plot_training_curve(train_losses, val_losses, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to: {save_path}")


def direct_evaluate(model, dataloader, device, run_dir):
    model.eval()
    preds_list = []
    labels_list = []
    probs_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating (Direct)"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].cpu().numpy()

            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predicted = torch.argmax(logits, dim=1).cpu().numpy()

            preds_list.extend(predicted)
            labels_list.extend(labels)
            probs_list.extend(probs)

    metrics_result = compute_metrics(
        labels=np.array(labels_list),
        preds=np.array(preds_list),
        probs=np.array(probs_list)
    )

    result_path = os.path.join(run_dir, "direct_evaluate_results.txt")
    with open(result_path, "w") as f:
        f.write(str(metrics_result))

    print(f"[Direct Evaluate] Results: {metrics_result}")
    print(f"File saved to: {result_path}")
    return metrics_result


def finetune(model, train_loader, val_loader, device, epochs, lr, run_dir, weight):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device))

    model.train()
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    best_model_path = os.path.join(run_dir, "best_ft_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch}/{epochs} -> Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model, best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f} at {best_model_path}")

    final_model_path = os.path.join(run_dir, "finetuned_model.pth")
    torch.save(model, final_model_path)
    print(f"Finetuning completed, model saved at: {final_model_path}")


def linear_probe(model, train_loader, val_loader, device, epochs, lr, run_dir, weight):
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    if weight is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight).to(device))
    model.train()
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    best_model_path = os.path.join(run_dir, "best_linear_probe_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Linear Probe Epoch {epoch}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits, _ = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_epoch_loss = val_loss / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"Epoch {epoch}/{epochs} -> Train Loss: {train_epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model, best_model_path)
            print(f"Best model saved with val loss: {best_val_loss:.4f} at {best_model_path}")

    final_model_path = os.path.join(run_dir, "linear_probe_model.pth")
    torch.save(model, final_model_path)
    print(f"Linear probe completed, model saved at: {final_model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="aft_ft",
                        choices=["direct", "finetune", "linear", "aft_ft"],
                        help="Choose the task to execute: direct, finetune, linear, aft_ft")
    parser.add_argument("--dev_tsv", type=str, default="gen_dean_role/deanrole_sentence/4_class_wo_split.tsv",
                        help="Path to the dataset file")
    parser.add_argument("--pretrained_path", type=str, default="Train_output/01_09_17_47", help="Pretrained model path")
    parser.add_argument("--model_path", type=str, default="Eval_deanrole/finetune/01_14_21_23/best_ft_model.pth",
                        help="Path to the model to evaluate/continue training")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="Max length for BERT input")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of classes")
    parser.add_argument("--lossfuc_weight", default=None, help="Loss function weight")
    args = parser.parse_args()

    if args.task not in ["direct", "finetune", "linear", "aft_ft"]:
        raise ValueError("Invalid task! Must be one of: direct, finetune, linear, aft_ft")

    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
    run_dir = f"Eval_deanrole/{args.task}/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    print(f"Running task: {args.task}")
    config_txt_path = os.path.join(run_dir, "config.txt")
    with open(config_txt_path, "w") as f:
        f.write("Running Configurations:\n")
        f.write(str(args) + "\n")

    device_ids = [1, 2, 3]
    torch.cuda.set_device(device_ids[0])

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_path)

    dataset = DevDataset(tsv_file=args.dev_tsv, tokenizer=tokenizer, max_len=args.max_length)
    print(f"Loaded dataset from {args.dev_tsv}, total samples: {len(dataset)}")
    with open(config_txt_path, "a") as f:
        f.write(f"Loaded dataset from {args.dev_tsv}, total samples: {len(dataset)}\n")

    train_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    freeze_bert_flag = True if args.task == "linear" else False
    model = BertClassifier(
        pretrained_model_path=args.pretrained_path,
        num_labels=args.num_labels,
        freeze_bert=freeze_bert_flag
    )
    print(f"Model loaded, freeze_bert={freeze_bert_flag}")
    with open(config_txt_path, "a") as f:
        f.write(f"Model loaded. freeze_bert={freeze_bert_flag}\n")

    if len(device_ids) > 1:
        print(f"Using GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.to(device_ids[0])

    if args.task == "direct":
        model.eval()
        direct_metrics = direct_evaluate(model, val_loader, device_ids[0], run_dir)
        print(f"Direct evaluate results: {direct_metrics}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Direct Evaluate Results:\n")
            f.write(str(direct_metrics) + "\n")

    elif args.task == "aft_ft":
        loaded_model = torch.load(args.model_path)
        loaded_model.eval()
        loaded_model.to(device_ids[0])

        direct_metrics = direct_evaluate(loaded_model, val_loader, device_ids[0], run_dir)
        print(f"Evaluate results after supervised training: {direct_metrics}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Evaluate Results After Supervised Training:\n")
            f.write(str(direct_metrics) + "\n")

    elif args.task == "finetune":
        finetune(model, train_loader, val_loader, device_ids[0],
                 epochs=args.epochs,
                 lr=args.learning_rate,
                 run_dir=run_dir,
                 weight=args.lossfuc_weight)

        model.eval()
        preds_list = []
        labels_list = []
        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device_ids[0])
                attention_mask = batch['attention_mask'].to(device_ids[0])
                labels = batch['label'].cpu().numpy()

                logits, _ = model(input_ids, attention_mask)
                prob_batch = torch.softmax(logits, dim=1).cpu().numpy()
                preds_batch = torch.argmax(logits, dim=1).cpu().numpy()

                preds_list.extend(preds_batch)
                labels_list.extend(labels)
                probs_list.extend(prob_batch)

        metrics_result = compute_metrics(np.array(labels_list), np.array(preds_list), np.array(probs_list))
        print(f"[Finetune Evaluate] {metrics_result}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Finetune Evaluate Results:\n")
            f.write(str(metrics_result) + "\n")

    elif args.task == "linear":
        linear_probe(model, train_loader, val_loader, device_ids[0],
                     epochs=args.epochs,
                     lr=1e-3,
                     run_dir=run_dir,
                     weight=args.lossfuc_weight)

        model.eval()
        preds_list = []
        labels_list = []
        probs_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device_ids[0])
                attention_mask = batch['attention_mask'].to(device_ids[0])
                labels = batch['label'].cpu().numpy()

                logits, _ = model(input_ids, attention_mask)
                prob_batch = torch.softmax(logits, dim=1).cpu().numpy()
                preds_batch = torch.argmax(logits, dim=1).cpu().numpy()

                preds_list.extend(preds_batch)
                labels_list.extend(labels)
                probs_list.extend(prob_batch)

        metrics_result = compute_metrics(np.array(labels_list), np.array(preds_list), np.array(probs_list))
        print(f"[Linear Probe Evaluate] {metrics_result}")

        result_txt_path = os.path.join(run_dir, "result.txt")
        with open(result_txt_path, "w") as f:
            f.write("Linear Probe Evaluate Results:\n")
            f.write(str(metrics_result) + "\n")


if __name__ == "__main__":
    main()