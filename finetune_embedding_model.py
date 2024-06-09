from question_answering_dataset_generator import random_texts, questions_generation
from sentence_transformers import InputExample
from llama_index import StorageContext, load_index_from_storage

from torch.utils.data import DataLoader
import random

import joblib
from sentence_transformers import losses, SentenceTransformer
import torch
from tqdm import tqdm


path_vector_db = "/Users/jean-baptistechaudron/Documents/Uruk/<uruk_maquette_vicuna>"

storage_context = StorageContext.from_defaults(persist_dir=path_vector_db)
index = load_index_from_storage(storage_context)

answers = random_texts(index, n_texts=100)
dataset = questions_generation(answers)

all_data = list(dataset.items())
train_index = random.sample(list(range(len(all_data))), k = int(0.7*len(all_data)))
train_data = [all_data[i] for i in train_index]
eval_data = [all_data[i] for i in range(len(all_data)) if not i in train_index]

joblib.dump(train_data,"training_qa_dataset.joblib")
joblib.dump(eval_data,"evaluation_qa_dataset.joblib")

train_examples = []

for (q,a) in train_data:
    train_examples.append(InputExample(texts=[q,a]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1-finetuned"
model = SentenceTransformer(embedding_model)
model.to("mps")
train_loss = losses.MultipleNegativesRankingLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
model.save()

result = []

model.eval()
for (q,a) in tqdm(eval_data):
    with torch.no_grad():
        tokenized_q, tokenized_a = model.tokenize(q), model.tokenize(a)
        tokenized_q = {key : item.to("mps") for key, item in tokenized_q.items()}
        tokenized_a = {key : item.to("mps") for key, item in tokenized_a.items()}
        embedding_q, embedding_a = model(tokenized_q), model(tokenized_a)
        loss = torch.dot(embedding_q["sentence_embedding"].mean(0),embedding_a["sentence_embedding"].mean(0))
        print(loss)
        result.append(loss.detach().tolist())


vanilla_embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model_vanilla = SentenceTransformer(embedding_model)
model_vanilla.eval()
results_vanilla = []
for (q,a) in tqdm(eval_data):
    with torch.no_grad():
        tokenized_q, tokenized_a = model_vanilla.tokenize(q), model_vanilla.tokenize(a)
        tokenized_q = {key : item.to("mps") for key, item in tokenized_q.items()}
        tokenized_a = {key : item.to("mps") for key, item in tokenized_a.items()}
        embedding_q, embedding_a = model_vanilla(tokenized_q), model_vanilla(tokenized_a)
        loss = torch.dot(embedding_q["sentence_embedding"].mean(0),embedding_a["sentence_embedding"].mean(0))
        print(loss)
        results_vanilla.append(loss.detach().tolist())