from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation, concatenate, multiply, MaxPooling1D, Conv1D, Flatten
from keras.optimizers import RMSprop
import keras
import tensorflow as tf

from .myModels.attendgru import AttentionGRUModel
from .myModels.ast_attendgru import AstAttentionGRUModel
#from models.pretrained import Pretrained
#from models.vae_like import VAELike
#from models.vae_like_ast import VAELikeAST
#from models.vae_lstm_ast import VAELikeASTLSTM
#from models.vae_astattendgru import VAEASTAtt
#from models.vae_astattendgru300 import VAEASTAtt300
from .myModels.ast_attendgru_xtra import AstAttentionGRUModel as xtra
def create_model(modeltype, config):
    mdl = None

    if modeltype == 'attend-gru':
    	# base attention GRU model based on Nematus architecture
        mdl = AttentionGRUModel(config)
    elif modeltype == 'ast-attendgru':
    	# attention GRU model with added AST information from srcml. 
        mdl = AstAttentionGRUModel(config)
    elif modeltype == 'pretrained':
    	# ast-attendgru with pretrained word embeddings for dats/coms similar to codecat.
        mdl = Pretrained(config)
    elif modeltype == 'vae':
        mdl = VAELike(config)
    elif modeltype == 'vae-ast':
        mdl = VAELikeAST(config)
    elif modeltype == 'vae-lstm-ast':
        mdl = VAELikeASTLSTM(config)
    elif modeltype == 'vae-astatt':
        mdl = VAEASTAtt(config)
    elif modeltype == 'vae-astatt300':
        mdl = VAEASTAtt300(config)
    elif modeltype == 'xtra':
        mdl = xtra(config)
    else:
        print("{} is not a valid model type".format(modeltype))
        exit(1)
        
    return mdl.create_model()
