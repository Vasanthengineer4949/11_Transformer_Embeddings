import config
from embedder import Embedder
from model import Model
import pandas as pd

if __name__ == "__main__":
    data_embedder = Embedder()
    data_embedder.run(config.DATA_ID, config.MODEL_CKPT, config.EMBEDDINGS_PATH)
    data = pd.read_csv(config.EMBEDDINGS_PATH)
    modeller = Model(data)
    modeller.run()

