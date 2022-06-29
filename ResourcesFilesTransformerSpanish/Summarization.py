import pandas as pd 
import numpy as np
import pickle
###Librerias de transformador
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    PreTrainedTokenizer
)
import torch
import tensorflow as tf
import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import string
from Transformador.transformadorPreprocesamiento import Preprocesamiento
import tqdm as notebook_tqdm
from Transformer import TextSummarizationPredict

class SummarizationPredict:
    """ Clase TextSummarizationPredict """
    
    def __init__(self) -> None:
        """ Inicia la clase TextSummarizationT5 """
        pass
   
    def cargarPipeline(self, nombreArchivo):
        with open(nombreArchivo+'.pickle', 'rb') as handle:
            pipeline = pickle.load(handle)
        return pipeline
    
    def limpiezaTextoParaPredict(self, texto):
        pipe = self.cargarPipeline('Transformador/pipePreprocesador')
        text = texto
        textProcesado = pipe.transform(text)
        return textProcesado['cleaned_text'][0]
    
    def predictSummarization(self, texto):
        model = TextSummarizationPredict()
        model.load_model("t5","modelo/TextSummarizationT5/TextSummarizationT5-epoch-7-train-loss-0.4759-val-loss-1.2537", use_gpu=False)
        text_to_summarize=self.limpiezaTextoParaPredict(texto)
        return(model.predict(text_to_summarize))
        
            