# To transform the documents into "nodes" form

from llama_index.node_parser import SimpleNodeParser

from llama_index import StorageContext, load_index_from_storage
from llama_index import Prompt

from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context, EmptyIndex
from llama_index.llms import LlamaCPP
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm




import random
def random_texts(index_in, n_texts=100):
    docstore = index_in.docstore.docs
    all_nodes = list(docstore.keys())
    text_out = [node for node in random.choices(all_nodes,k=n_texts)]
    return [docstore[text_id].text for text_id in text_out]


def questions_generation(answers, llm):
    llm = llm
    out = {}
    for answer in tqdm(answers):
        prompt = """
        Donne des questions auquel le texte suivant pourrait répondre
        {}
        Énumère des questions auxquelles le texte précédent pouvait répondre, sépare les par un retour à la ligne
        Répond en français !
        Chaque questions doit être auto suffisante et ne pas faire référence à une autre question précédemment vue
        """.format(answer)
        try:
            questions = llm.complete(prompt)
            questions = questions.text.split("\n")
            out.update({question : answer for question in questions})
        except Exception:
            pass
    return out

