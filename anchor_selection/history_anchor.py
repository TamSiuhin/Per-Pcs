from transformers import DebertaV2Tokenizer, DebertaV2Model
import torch
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from sklearn.cluster import KMeans
import numpy as np

batch_size = 32

def get_first_k_tokens(text, k):
    """
    Extracts the first k tokens from a text string.

    :param text: The input text string.
    :param k: The number of tokens to extract.
    :return: The first k tokens of the text string.
    """
    # Split the text into tokens based on whitespace
    tokens = text.split()
    output = " ".join(tokens[:k])

    # Return the first k tokens
    return output

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def extract_article(text):
    marker = "] description: "
    # Find the position of the marker in the text
    marker_pos = text.find(marker)
    
    # Check if the marker is found
    if marker_pos == -1:
        raise ValueError()

    # Extract the string after the marker
    extracted_string = text[marker_pos + len(marker):]

    return extracted_string

parser = argparse.ArgumentParser("MAE pre-training", add_help=False)
parser.add_argument("--candidate_path", default="../data/movie_tagging/user_anchor_candidate.json", type=str, help="path to candidate user json file")
parser.add_argument("--task_name", default="movie_tagging", type=str, metavar="MODEL", help="name of the task")
parser.add_argument("--k", default=50, type=int,help="number of selected anchor user")

args = parser.parse_args()


with open(args.candidate_path, 'r') as f:
    anchor_candidate = json.load(f)

with open('../prompt/prompt.json', 'r') as f:
    prompt_template = json.load(f)

# Step 1: Load the DeBERTa-v3-base tokenizer and model
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-large')
model = DebertaV2Model.from_pretrained('microsoft/deberta-v3-large').cuda()


all_user_emb = []
for user in tqdm(anchor_candidate):

    history_embeddings_list = []

    visible_history_list = user['profile']
    for p in visible_history_list:
        for key, value in p.items():
            p[key] = get_first_k_tokens(p[key], 368)

    user_nl_history_list = [prompt_template[args.task_name]['retrieval_history'].format(**p) for p in visible_history_list]

    user_nl_history_list_batched = split_batch(user_nl_history_list, batch_size)

    for batch in tqdm(user_nl_history_list_batched):

        with torch.no_grad():
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs)

            last_hidden_states = outputs.last_hidden_state
            # Compute attention mask
            attention_mask = inputs['attention_mask']

            # Expand attention mask so it has same size as last_hidden_states, for broadcasting purposes
            attention_mask = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()

            # Multiply last hidden states by attention mask, then sum and divide by number of tokens
            masked_hidden_states = last_hidden_states * attention_mask
            summed = torch.sum(masked_hidden_states, 1)
            count = torch.clamp(attention_mask.sum(1), min=1e-9)
            mean_pooled = summed / count

        history_embeddings_list.append(mean_pooled.cpu())

    history_embedding_concat = torch.cat(history_embeddings_list, dim=0).cpu().mean(dim=0, keepdim=True)
    all_user_emb.append(history_embedding_concat)

all_user_emb = torch.cat(all_user_emb, dim=0)
print(all_user_emb.size())

Path(f'./{args.task_name}/').mkdir(parents=True, exist_ok=True)

torch.save(all_user_emb, f'./{args.task_name}/user_history_emb.pt')


emb = all_user_emb.numpy()

k=args.k
kmeans = KMeans(n_clusters=k, random_state=0, max_iter=3000).fit(emb)
labels = kmeans.labels_

selected_indices = []

for i in range(k):
    cluster_indices = np.where(labels == i)[0]
    max_len = 0
    for idx in cluster_indices:
        if len(anchor_candidate[idx]['profile']) > max_len:
            max_len = len(anchor_candidate[idx]['profile'])
            selected_index = idx
    print(max_len)

    if max_len>10:
        # selected_index = random.choice(cluster_indices)
        selected_indices.append(selected_index)

print(len(selected_indices))

torch.save(selected_indices, f'./{args.task_name}/anchor_user_idx.pt')

print('Done!')
