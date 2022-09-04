import config
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

class Embedder():

    def __init__(self):
        pass

    def load_default_data(self, data_id, splitby="train"):
        data = load_dataset(data_id, split=splitby)
        data.set_format("pandas")
        df = data[:]
        return df
        
    def embedding_generator(self, df, model):
        tqdm.pandas()
        model = SentenceTransformer(model)
        df['Embeddings'] = df['text'].progress_apply(lambda x: model.encode(x))
        return df

    def inputgen(self, embedding_generation_out):
        a = []
        for i in tqdm(range(len(embedding_generation_out))):
            a.append(np.array(embedding_generation_out["Embeddings"][i]))
        a = np.array(a)
        print(a.shape)
        return a

    def output_emb_df(self, df, inputgen_out):
        embed = pd.DataFrame(inputgen_out)
        df = pd.concat([df, embed], axis=1)
        return df

    def run(self, data_id, model, out_path):
        data = self.load_default_data(data_id)
        print("Loaded data")
        embeddings = self.embedding_generator(data, model)
        print("Embeddings Generated")
        embeddings_df = self.inputgen(embeddings)
        print("Embeddings Dataframe Created")
        final_csv = self.output_emb_df(data, embeddings_df)
        print("Final CSV obtained")
        final_csv.to_csv(out_path, index=False)
        print("CSV file saved")

