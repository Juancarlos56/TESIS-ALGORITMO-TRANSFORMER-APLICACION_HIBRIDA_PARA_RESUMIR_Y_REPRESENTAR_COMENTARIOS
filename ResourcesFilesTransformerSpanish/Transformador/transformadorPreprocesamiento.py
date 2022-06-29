from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Transformador import preprocesamientoTexto
##Comencemos creando un DummyEstimator, del cual heredaremos init, fit y transform. 
##DummyEstimator es una clase útil que nos evita escribir código redundante.

class DummyTransformer(BaseEstimator, TransformerMixin):
    """
      Dummy class that allows us to modify only the methods that interest us,
      avoiding redudancy.
    """
    def __init__(self):
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        return self

class Preprocesamiento(DummyTransformer):
    
    def transform(self, X=None):
        dataset = preprocesamientoTexto.TextoToDataframe(X)
        dataset = preprocesamientoTexto.eliminarValoresNulos(dataset)
        dataset = preprocesamientoTexto.limpiarNewSample(dataset)
        return dataset

if __name__=='__main__':
       print("Transformador cargado y listo...")