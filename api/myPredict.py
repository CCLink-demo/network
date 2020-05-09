import os
import sys
import traceback
import pickle
import argparse
import collections
from keras import metrics
import random
import tensorflow as tf
import numpy as np
from rake_nltk import Rake
from .apriori import apriori
import json

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import multiprocessing
from itertools import product

from multiprocessing import Pool

from timeit import default_timer as timer

from .model import create_model
from .myutils import prep, drop, statusout, batch_gen, seq2sent, index2word, init_tf, seq2word, seqList2Sent
import keras

from .myModels.attendgru import top2, top3, top5
from .myTokenizer import Token, Keyword


def top2(y1, y2):
    return metrics.top_k_categorical_accuracy(y1, y2, k=2)

def top3(y1, y2):
    return metrics.top_k_categorical_accuracy(y1, y2, k=3)

def top5(y1, y2):
    return metrics.top_k_categorical_accuracy(y1, y2, k=5)

def gendescr_3inp(model, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...
    
    dats, coms, smls = list(zip(*data.values()))
    dats = np.array(dats)
    coms = np.array(coms)
    smls = np.array(smls)

    for i in range(1, comlen):
        results = model.predict([dats, coms, smls], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data

def gendescr_2inp(model, data, comstok, comlen, batchsize, strat='greedy'):
    # right now, only greedy search is supported...
    
    dats, coms = list(zip(*data.values()))
    dats = np.array(dats)
    coms = np.array(coms)
    print(dats.shape)
    print(coms.shape)
    print(batchsize)
    exit()

    for i in range(1, comlen):
        results = model.predict([dats, coms], batch_size=batchsize)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)

    final_data = {}
    for fid, com in zip(data.keys(), coms):
        final_data[fid] = seq2sent(com, comstok)

    return final_data


def load_model_from_weights(modelpath, modeltype, datvocabsize, comvocabsize, smlvocabsize, datlen, comlen, smllen):
    config = dict()
    config['datvocabsize'] = datvocabsize
    config['comvocabsize'] = comvocabsize
    config['datlen'] = datlen # length of the data
    config['comlen'] = comlen # comlen sent us in workunits
    config['smlvocabsize'] = smlvocabsize
    config['smllen'] = smllen

    model = create_model(modeltype, config)
    model.load_weights(modelpath)
    return model

  
outdir = 'outdir'
dataprep = 'funcom/data/standard'
modelfile = 'funcom/standard/outdir/standard_attend-gru_E04_TA0.72_VA0.66.h5'
nmodelfile = 'funcom/standard/outdir/standard_xtra_E05_TA0.74_VA0.66.h5'
numprocs = '4'
gpu = '0'
modeltype = None
outfile = None

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(dataprep)
import tokenizer

datstok = pickle.load(open('funcom/data/standard/dats.tok', 'rb'), encoding='UTF-8')
comstok = pickle.load(open('funcom/data/standard/coms.tok', 'rb'), encoding='UTF-8')
smltok = pickle.load(open('funcom/data/standard/smls.tok', 'rb'), encoding='utf-8')

# model = keras.models.load_model(modelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})

comlen = 13
comstart = np.zeros(comlen)
st = comstok.w2i['<s>']
comstart[0] = st



def translateBatch(codeList, model, sbt=False, _sml=None):
    inpt = np.zeros((len(codeList), 100))
    for i, code in enumerate(codeList):
        for j, w in enumerate(code.split(' ')):
            if j == 100:
                break
            if w not in datstok.w2i.keys():
                inpt[i][j] = 0
            else:
                inpt[i][j] = datstok.w2i[w]
    coms = np.zeros((len(codeList), comlen))
    coms[:, 0] = st
    
    for i in range(1, comlen):
        if not sbt:
            results = model.predict([inpt, coms], batch_size=len(codeList))
        else:
            sml = np.zeros((len(codeList), 100))
            sml[:, :] = np.array(_sml)
            results = model.predict([inpt, coms, sml], batch_size=len(codeList))
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
    strComs = [i.split(' ') for i in seqList2Sent(coms, comstok)]
    transferedComs = []
    for com in strComs:
        s = ''
        for j in range(1, len(com)):
            if com[j] == '</s>':
                break
            s += com[j] + " "
        transferedComs.append(s)
    return [i.split(' ')[: -1] for i in transferedComs]
    

def translate(code, model, sbt=False, sml=None):
    words = code.split(' ')
    inpt = [np.zeros(100)]
    for i, w in enumerate(words):
        if i >= 100:
            break
        if w not in datstok.w2i.keys():
            inpt[0][i] = 0
        else:
            inpt[0][i] = datstok.w2i[w]
    coms = np.zeros(comlen)
    coms[0] = st
    coms = [coms]
        
    for i in range(1, comlen):
        if not sbt:
            results = model.predict([inpt, coms], batch_size=1)
        else:
            results = model.predict([inpt, coms, sml], batch_size=1)
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
    return seq2sent(coms[0], comstok).split(' ')

# key = 'sets'

def predictor(codeList):
    print(key)
    retList = []
    inpts = np.zeros((len(codeList), 100))
    coms = np.zeros((len(codeList), comlen))

    coms[:,0] = st
    # print(coms)

    for i, c in enumerate(codeList):
        for j, w in enumerate(c.split(' ')):
            if w not in datstok.w2i.keys():
                inpts[i][j] = 0
            else:
                inpts[i][j] = datstok.w2i[w]

    for i in range(1, comlen):
        results = model.predict([inpts, coms], batch_size=len(codeList))
        for c, s in enumerate(results):
            coms[c][i] = np.argmax(s)
    
    for c in coms:
        # print(seq2sent(c, comstok).split(' '))
        if key in seq2sent(c, comstok).split(' '):
            retList.append(1)
        else:
            retList.append(0)
    return np.array(retList)

def predict(codeList):
    retList = []
    for c in codeList:
        com = translate(c)
        print(com)
        if key in com:
            retList.append(1)
        else:
            retList.append(0)
    return retList

def translateStrs(code, model, sbt=False, sml=None):
    com = ''
    inp = ' '.join(code)
    c = translate(inp, model, sbt, sml)
    for i in range(1, len(c)):
        if c[i] == '</s>':
            break
        com += c[i] + ' '
    return com, c


def explain(code, key):
    for i in range(len(code)):
        for j in range(0, len(code) - i):
            print(j, j+i+1)
            com, _ = translateStrs(code[j: j+i+1])
            if key in com:
                return code[j: j+i+1], j, j+i+1

def putTogether(code):
    tmpList = []
    for l in code:
        tmpList.append(' '.join(l))
    return ' '.join(tmpList)


def generateCode(_code, key, ids):
    code = [i.copy() for i in _code]
    # print(code)
    for id in ids:
        codeKey = key[id]
        # print(codeKey, code)
        lineNum = codeKey[0]
        words = codeKey[1]
        for i in words:
            code[lineNum][i] = '<unk>'
    retCode = ''
    for l in code:
        retCode += ' '.join(l) + ' '
    return retCode


def explain_key(code, key, numSamples, comkey, model, sbt=False, sml=None, thresh=0.6):
    p = 0.1
    while p <= 1.0:
        i, ni = explain_key_once(code, key, numSamples, comkey, p, model, sbt, sml, thresh)
        if len(ni) >= numSamples:
            break
        else:
            p += 0.05
        # print(p, len(i), len(ni))
    return i, ni, p


def explain_key_once(code, key, numSamples, comkey, p, model, sbt=False, sml=None, thresh=0.6):
    codes = []
    keyIds = []
    inComs = []
    notInComs = []
    for i in range(int(numSamples / thresh)):
        numChanged = np.random.binomial(len(key), p)
        changed = np.random.choice(len(key), numChanged)
        keyIds.append(changed)
        codes.append(generateCode(code, key, changed))
    coms = translateBatch(codes, model, sbt, sml)
    for i, com in enumerate(coms):
        if comkey in com:
            inComs.append(keyIds[i])
        else:
            notInComs.append(keyIds[i])
    return inComs, notInComs


def explain_block(code, key):
    inComs = []
    notInComs = []

    for i in range(len(code)):
        for j in range(0, len(code) - i):
            temp = code[j: j+i+1]
            blockLines = []
            for l in temp:
                words = l.split(' ')
                t = ['<NULL>' for w in words]
                t = ' '.join(t)
                blockLines.append(t)
            tempCode = code.copy()
            tempCode[j: j+i+1] = blockLines
            com, _ = translateStrs(tempCode)
            if key in com:
                inComs.append((j, j+i+1))
            else:
                notInComs.append((j, j+i+1))
                
    return inComs, notInComs

def takeSecond(ele):
    return ele[0]


def getFromSeqData(id, seqData, datstok):
    key = list(seqData['dtest'].keys())[id]
    code = ''
    val = seqData['dtest'][key]
    for i in val:
        code += datstok.i2w[i] + ' '
    sml = seqData['stest'][key]
    return code, smlf


def getBatchCodeFile(filename):
    f = open(filename)
    codes = []
    for line in f.readlines():
        codes.append(line)
    return codes


def finalExplain(code, mode, sml=None):
    keras.backend.clear_session()
    retData = {}
    if mode == 'sbt':
        model = keras.models.load_model(nmodelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
        sbt = True
    else:
        model = keras.models.load_model(modelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
        sbt = False
    token = Token()
    
    code = [code.split('\n')]

    # code, lineNum = token.getLinesFromRawCodes([code])
    # print(1, code)
    # print(2, lineNum)
    code, lineNums = token.getFromFrontCode(code)
    print(code)
    print(lineNums)
    code = code[0]
    com, c = translateStrs(code, model, sbt, sml)
    retData['code'] = code
    retData['comment'] = com
    
    r = Rake()
    r.extract_keywords_from_text(com)
    comKeys = r.get_ranked_phrases()
    retData['commentKeywords'] = comKeys

    codeWordList = token.toDoubleList(code)
    codeKeys = []
    codeKeyIndex = []
    k = Keyword()
    for i, l in enumerate(code):
        r.extract_keywords_from_text(l)
        tmpKeys = r.get_ranked_phrases()
        t, filteredKey = k.preprocess(tmpKeys, i, l)
        codeKeyIndex += t
        codeKeys += filteredKey
    
    retIndex = [i for i, j in codeKeyIndex]
    retData['codeKeywordIndexs'] = retIndex
    
    retData['codeKeywords'] = codeKeys
    retData['codeKeywordDetailedIndex'] = codeKeyIndex

    tmpList = []
    for key in comKeys:
        tmpResults = {
            'commentKeyword': key,
        }
        keyNums = np.zeros(len(codeKeyIndex))
        i, ni, p = explain_key(codeWordList, codeKeyIndex, 100, key, model, sbt, sml, 0.6)
        tmpResults['numberHaveKey'] = len(i)
        tmpResults['numberNoKey'] = len(ni)
        tmpResults['probability'] = p

        for keyIds in ni:
            tmp = list(set(keyIds))
            for id in tmp:
                keyNums[id] += 1
        L, support = apriori(ni, 0.3)
        L = [[[int(j) for j in i] for i in l] for l in L]
        support = [[[int(i) for i in s[0]], s[1]] for s in support.items()]
        tmpResults['anchors'] = L
        tmpResults['supports'] = support

        tmpList.append(tmpResults)
    retData['explanations'] = tmpList
    del model
    return retData


def finalExplain_n(code, mode, model, sml=None):
    retData = {}
    token = Token()
    sbt = False
    
    code = [code.split('\n')]

    # code, lineNum = token.getLinesFromRawCodes([code])
    # print(1, code)
    # print(2, lineNum)
    code, lineNums = token.getFromFrontCode(code[0])
    print(code)
    print(lineNums)
    # code = code[0]
    com, c = translateStrs(code, model, sbt, sml)
    retData['code'] = code
    retData['comment'] = com
    
    r = Rake()
    r.extract_keywords_from_text(com)
    comKeys = r.get_ranked_phrases()
    retData['commentKeywords'] = comKeys

    codeWordList = token.toDoubleList(code)
    codeKeys = []
    codeKeyIndex = []
    k = Keyword()
    for i, l in enumerate(code):
        r.extract_keywords_from_text(l)
        tmpKeys = r.get_ranked_phrases()
        t, filteredKey = k.preprocess(tmpKeys, i, l)
        codeKeyIndex += t
        codeKeys += filteredKey
    
    retIndex = [i for i, j in codeKeyIndex]
    retData['codeKeywordIndexs'] = retIndex
    
    retData['codeKeywords'] = codeKeys
    retData['codeKeywordDetailedIndex'] = codeKeyIndex

    tmpList = []
    for key in comKeys:
        tmpResults = {
            'commentKeyword': key,
        }
        keyNums = np.zeros(len(codeKeyIndex))
        i, ni, p = explain_key(codeWordList, codeKeyIndex, 100, key, model, sbt, sml, 0.6)
        tmpResults['numberHaveKey'] = len(i)
        tmpResults['numberNoKey'] = len(ni)
        tmpResults['probability'] = p

        for keyIds in ni:
            tmp = list(set(keyIds))
            for id in tmp:
                keyNums[id] += 1
        L, support = apriori(ni, 0.3)
        L = [[[int(j) for j in i] for i in l] for l in L]
        support = [[[int(i) for i in s[0]], s[1]] for s in support.items()]
        tmpResults['anchors'] = L
        tmpResults['supports'] = support

        tmpList.append(tmpResults)
    retData['explanations'] = tmpList
    del model
    return retData


if __name__ == '__main__':
    code = '''public void disconnect() {
        try {
            socket.flush();
            socket.close();
            connected = false;
            notifyDisconnect();
        } catch (IOException ex) {
        ex.printStackTrace(); } }
    '''
    exp = finalExplain(code, 'normal', None)
    print(exp)
    exit()

    # from tokenize import Token
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('file', type=str, default='test.java')
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--batch', action='store_true', default=False)
    parser.add_argument('--num', dest='num', type=int, default=-1)
    # parser.add_argument('--dataset', dest='dataSet', type=str, default=None)
    args = parser.parse_args()
    sbt = args.sbt
    batch = args.batch
    num = args.num
    saveData = {}
    # dataSet = args.dataSet
    if batch:
        rawCodes = getBatchCodeFile(args.file)
        if num < 0:
            num = len(rawCodes)
        rawCodes = rawCodes[: num]
        model = keras.models.load_model(modelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
    elif not sbt:
        codeFile = args.file
        model = keras.models.load_model(modelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
    else:
        seqdata = pickle.load(open('funcom/data/standard/dataset.pkl', 'rb'))
        codeFile = int(args.file)
        model = keras.models.load_model(nmodelfile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})

    token = Token()
    if batch:
        print('have ' + str(num) + ' pieces of codes')
        codes = token.getLinesFromRawCodes(rawCodes)
        # print(len(codes))
        codes = token.getFromLines(codes)
        # print(codes[0])
        sml = None
    elif not sbt:
        code = token.getFromFile(codeFile)
        codes = [code]
        sml = None
    else:
        code, sml = getFromSeqData(codeFile, seqdata, datstok)
        code = [code[:-1]]
        codes = [code]
        sml = [sml]
    
    if batch:
        print(codes[0])
    else:
        print(code)
        print(num)
    # print(sml)
    for m in range(num):
        tmpData = {}
        _code = codes[m]
        print(1, _code)
        com, c = translateStrs(_code, model, sbt, sml)

        if batch:
            tmpData['code'] = rawCodes[m]
        else:
            tmpData['code'] = code
        tmpData['comment'] = com
        # print(com)
        # print(c)
        print()
    
        r = Rake()
        r.extract_keywords_from_text(com)
        keys = r.get_ranked_phrases()
        tmpData['commentKeywords'] = keys

        codeWordList = token.toDoubleList(_code)
        # print(len(codeWordList[0]))

        codeKeys = []
        codeKeyIndex = []
        k = Keyword()
        for i, l in enumerate(_code):
            r.extract_keywords_from_text(l)
            tmpKeys = r.get_ranked_phrases()
            t, filteredKey = k.preprocess(tmpKeys, i, l)
            codeKeyIndex += t
            codeKeys += filteredKey

        # codeKeysList = k.prepocess(codeKeys, )
        # print(codeKeys)
        tmpData['codeKeywords']= codeKeys
        # print(codeKeyIndex)
        # print(len(codeKeyIndex))

        tmpList = []

        for key in keys:
            tmpResults = {
                'commentKeyword': key,
            }
            keyNums = np.zeros(len(codeKeyIndex))
            # print('key: {}'.format(key))
            i, ni, p = explain_key(codeWordList, codeKeyIndex, 100, key, model, sbt, sml, 0.6)
            tmpResults['numberHaveKey'] = len(i)
            tmpResults['numberNoKey'] = len(ni)
            tmpResults['probability'] = p
            # print('number have key: {}'.format(len(i)))
            # print('number have no key: {}'.format(len(ni)))
            for keyIds in ni:
                tmp = list(set(keyIds))
                for id in tmp:
                    keyNums[id] += 1
            L, support = apriori(ni, 0.3)
            print(L)
            L = [[[int(j) for j in i] for i in l] for l in L]
            support = [[[int(i) for i in s[0]], s[1]] for s in support.items()]
            print(support)
            tmpResults['anchors'] = L
            tmpResults['supports'] = support
        
            if len(L) == 1:
                # print(keyNums)
                # print()
                continue
            for items in L[-2]:
                s = ''
                for i in items:
                    s += codeKeys[i] + '; '
                # print(s)
            # print(keyNums)
            # print()
            tmpList.append(tmpResults)
        tmpData['explanations'] = tmpList
        saveData[m] = tmpData
    # print(saveData)
    with open('testOut.json', 'w') as f:
        json.dump(saveData, f, indent=2)


    # for key in keys:
    #     lineNum = np.zeros(len(code))
    #     print("key: {}".format(key))
    #     i, ni = explain_block(code, key)
    #     print('number have key: {}'.format(len(i)))
    #     print('number no key: {}'.format(len(ni)))
    #     for i, j in ni:
    #         lineNum[i:j] += 1
    #     print(lineNum)
    #     print()