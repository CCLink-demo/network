from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.settings import api_settings
import json
import requests
import keras
from .model import create_model
import os
import time
from django.conf import settings
# from api.models import exp

outdir = 'outdir'
dataprep = ''

# modify to your own model file
modelfile = ''

numprocs = '4'
gpu = '0'
modeltype = None
outfile = None

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .myPredict import finalExplain, top2, top3, top5, finalExplain_n

# Create your views here.

import tensorflow as tf
from .Exp import Explain

graph = tf.get_default_graph()
model = None
model = keras.models.load_model(modelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
mode = 1000
exp = Explain(_codeFile=None, _mode=1000, model=model)

ERROR_CODE = {
    'missingCriticalInformation': 1
}

def GenError(errorCode: int):
    if errorCode == 1:
        return {'state': 1, 'error': 'Missing Critical Information'}

class Explain(APIView):
    def post(self, request):
        try:
            _code = request.data.get('code')
            _mode = request.data.get('mode')
        except:
            return Response(GenError(ERROR_CODE['missingCriticalInformation']), status=status.HTTP_400_BAD_REQUEST)
        
        mode = 1000
        if _mode == 'sbt':
            # generate sbt
            sbt = ''
            mode = 1010
            
        # exp = finalExplain(_code, _mode)

        global graph
        with graph.as_default():
            # exp = Explain(_codeFile=_code, _mode=1000, model=model)
            exp.reload(_code, mode)
            res = exp.explain()
            # res = finalExplain_n(_code, _mode, model)
        return Response(res[0])

class SaveFile(APIView):
    def post(self, request):
        try:
            _userName = request.data.get('id')
            _commentSet = request.data.get('commentSet')
            _commentTimeSet = request.data.get('commentTimeSet')
            _second = request.data.get('second')
        except:
            return Response(GenError(ERROR_CODE['missingCriticalInformation']), status=status.HTTP_400_BAD_REQUEST)

        # print(_comments)
        saveCommentSet = {
            'id': _userName,
            'commentSet': _commentSet,
        }
        saveCommentTimeSet = {
            'id': _userName,
            'commentTimeSet': _commentTimeSet,
            'second': _second
        }
        ts = int(time.time())
        fileName = '{}_{}_{}.json'.format(_userName, str(ts), 'commentSet')
        filePath = os.path.join(settings.UPLOAD_DIR, fileName)
        with open(filePath, 'w') as f:
            json.dump(saveCommentSet, f, indent=2)
        fileName = '{}_{}_{}.json'.format(_userName, str(ts), 'commentTimeSet')
        filePath = os.path.join(settings.UPLOAD_DIR, fileName)
        with open(filePath, 'w') as f:
            json.dump(saveCommentTimeSet, f, indent=2)
        return Response({'state': 0})