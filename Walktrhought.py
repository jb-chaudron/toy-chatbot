from llama_index.node_parser import SimpleNodeParser
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context, EmptyIndex
from llama_index.llms import LlamaCPP
import os
from sentence_transformers import SentenceTransformer
from llama_index.node_parser import SimpleNodeParser

# J'utilise ce modèle qui est pas mal en multilingue, mais c'est peut-être pas l'état de l'art
# Tu peux quand même spécialiser le modèle pour tes données en utilisant le fichier finetuning
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot")

"""
================= #### En dessous la liste des paramètres que tu peux changer ### ===========

    1 - path_vector_db : le nom que tu veux donner à la base de données
    2 - Le chemin pour le modèle mistral, si tu l'as pas il se téléchargera tout seul sinon tu peux donner le chemin pour le modèle que tu veux (à noter que tu peux mettre n'importe quoi en .gguf comme modèle
        Si tu veux un autre mistral, tu peux changer la partie entre crochet je crois 
            mistral-7b-instruct-v0.1.{Q5_K_M}.gguf
        Mais la il est pas mal, si tu met Q plus bas tu auras un modèle plus comprésser pas top
        Moi ça marche sur mon mac avec 8GB de VRAM GPU
    
    3 - Le chemin des documents que tu veux analyser

    4 - Paramètre pour la création de la base de données
        CHUNK_SIZE : c'est la taille des morceaux de textes qui vont être produit
        CHUNK_OVERLAP : Deux morceaux consécutifs vont partager tant de tokens

"""
### 1 - Nom de la base de données
path_vector_db = "<vector_database>"

### 2 - Chemin pour mistral, change pas l'URL sauf si c'est pour prendre un autre modèle
model_url = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
path_model = "/path_to_the_model_mistral/mistral-7b-instruct-v0.1.Q5_K_M.gguf"


llm = LlamaCPP(model_url=model_url,
                model_path=path_model,
                model_kwargs={"n_gpu_layers": -1},
                verbose=True,
                temperature=0.2)

### 3 - Chemin des données
data_path = "/path_to_the_data/"

### 4 - Paramètres pour la création de la base de données
CHUNK_SIZE = 512 
CHUNK_OVERLAP = 32

### 5 - Extraction des données
documents = SimpleDirectoryReader(data_path).load_data()

parser = SimpleNodeParser.from_defaults()
nodes = parser.get_nodes_from_documents(documents)

### 6 - Définition des paramètres de plongement
service_context = ServiceContext.from_defaults(llm=llm,
                                               embed_model=embedding_model,
                                               chunk_size=512,
                                               chunk_overlap=32)
set_global_service_context(service_context)

index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )
index.storage_context.persist(persist_dir=path_vector_db)


