import os
import sys
import traceback
import pickle
import argparse
import collections
import random

from keras import metrics
import tensorflow as tf
import numpy as np

seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

from rake_nltk import Rake
from .apriori import apriori
import json

from .model import create_model
from .myutils import seq2sent, seq2word, seqList2Sent
import keras

from .myModels.attendgru import top2, top3, top5
from .myTokenizer import Token, Keyword
import progressbar

# modify the model file and token file
modelFile = ''
sbtModelFile = ''
datstokFile = ''
comstokFile = ''
smltokFile = ''

comlen = 13
comstart = np.zeros(comlen)

class Explain():
    def __init__(self, _codeFile, _mode, model=None):
        super().__init__()
        self.datstok = pickle.load(open(datstokFile, 'rb'), encoding='UTF-8')
        self.comstok = pickle.load(open(comstokFile, 'rb'), encoding='UTF-8')
        self.smltok = pickle.load(open(smltokFile, 'rb'), encoding='utf-8')
        self.model = model
        if model == None:
            self.model = self.loadModel(_mode)
        if _codeFile != None:
            self.rawData, self.num = self.loadData(_codeFile, _mode)
            self.tokenedCodes = self.tokenizeCodes(self.rawData, _mode)
        self.mode = _mode
        self.sml = None
        self.r = Rake()
        self.tokenizer = Token()
        self.st = self.comstok.w2i['<s>']
    
    def reload(self, _codeFile, _mode):
        self.rawData, self.num = self.loadData(_codeFile, _mode)
        self.tokenedCodes = self.tokenizeCodes(self.rawData, _mode)
    
    def reloadData(self, _codeFile, _mode):
        self.rawData, self.num = self.loadData(_codeFile, _mode)
        self.tokenedCodes = self.tokenizeCodes(self.rawData, _mode)
    
    def explain(self):
        print('start')
        retData = []
        # with progressbar.ProgressBar(self.num) as bar:
        for m in range(self.num):
            tmpData = {}
            code = self.tokenedCodes[m]
            com, c = self.translateStrs(code, self.checkMode(self.mode, 'withSbt'), self.sml)
            tmpData['code'] = code
            tmpData['comment'] = com

            self.r.extract_keywords_from_text(com)
            comKeys = self.r.get_ranked_phrases()
            tmpData['commentKeywords'] = comKeys

            codeWordList = self.tokenizer.toDoubleList(code)
            codeKeys, codeKeyIndex = self.extractCodeKeys(code)
            tmpData['codeKeywords'] = codeKeys
            tmpData['codeKeyIndex'] = codeKeyIndex
            tmpData['codeKeywordDetailedIndex'] = codeKeyIndex
            tmpList = []
            for key in comKeys:
                tmpResults = {
                    'commentKeyword': key,
                }
                keyNums = np.zeros(len(codeKeyIndex))
                i, ni, p = self.explainKey(codeWordList, codeKeyIndex, 100, key, 0.6)
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
            tmpData['explanations'] = tmpList
            retData.append(tmpData)
                # bar.update(m)
        return retData
    
    @staticmethod
    def ndarray2List(ndarray):
        retList = [int(i) for i in ndarray]
        return retList

    def explainLine(self):
        retData = []
        with progressbar.ProgressBar(self.num) as bar:
            for m in range(self.num):
                tmpData = {}
                code = self.tokenedCodes[m]
                com, c = self.translateStrs(code, self.checkMode(self.mode, 'withSbt'), self.sml)
                tmpData['code'] = code
                tmpData['comment'] = com

                self.r.extract_keywords_from_text(com)
                comKeys = self.r.get_ranked_phrases()
                tmpData['commentKeywords'] = comKeys

                tmpList = []
                for key in comKeys:
                    tmpResults = {
                        'commentKeyword': key,
                    }
                    lineNum = np.zeros(len(code))
                    i, ni = self.explainLineBase(code, key)
                    tmpResults['numberHaveKey'] = i
                    tmpResults['numberNoKey'] = ni
                    for i, j in ni:
                        lineNum[i: j] += 1
                    tmpResults['lineWeight'] = self.ndarray2List(lineNum)
                    tmpList.append(tmpResults)
                tmpData['explanations'] = tmpList
                retData.append(tmpData)
                bar.update(m)
        return retData

    def explainLineBase(self, code, key):
        inComs = []
        notInComs = []

        for i in range(len(code)):
            for j in range(0, len(code) - i):
                temp = code[j: j + i + 1]
                blockLines = []
                for l in temp:
                    words = l.split(' ')
                    t = ['<NULL>' for w in words]
                    t = ' '.join(t)
                    blockLines.append(t)
                tempCode = code.copy()
                tempCode[j: j + i + 1] = blockLines
                com, _ = self.translateStrs(tempCode, self.checkMode(self.mode, 'withSbt'), self.sml)
                if key in com:
                    inComs.append((j, j + i + 1))
                else:
                    notInComs.append((j, j + i + 1))
        return inComs, notInComs

    def explainKey(self, codeWordList, codeKeyIndex, numSamples, comKey, thresh=0.6):
        p = 0.1
        while p <= 1.0:
            i, ni = self.explainKeyOnce(codeWordList, codeKeyIndex, numSamples, comKey, p, thresh)
            if len(ni) >= numSamples:
                break
            else:
                p += 0.05
        return i, ni, p

    def explainKeyOnce(self, code, key, numSamples, comKey, p, thresh):
        codes = []
        keyIds = []
        inComs = []
        notInComs = []
        for i in range(int(numSamples / thresh)):
            numChanged = np.random.binomial(len(key), p)
            changed = np.random.choice(len(key), numChanged)
            keyIds.append(changed)
            codes.append(self.generateCode(code, key, changed))
        coms = self.translateBatch(codes)
        for i, com in enumerate(coms):
            if comKey in com:
                inComs.append(keyIds[i])
            else:
                notInComs.append(keyIds[i])
        return inComs, notInComs
    
    @staticmethod
    def generateCode(_code, key, ids):
        code = [i.copy() for i in _code]
        for id in ids:
            codeKey = key[id]
            lineNum = codeKey[0]
            words = codeKey[1]
            for i in words:
                code[lineNum][i] = '<unk>'
        retCode = ''
        for l in code:
            retCode += ' '.join(l) + ' '
        return retCode

    def extractCodeKeys(self, code):
        codeKeys = []
        codeKeyIndex = []
        k = Keyword()
        for i, l in enumerate(code):
            self.r.extract_keywords_from_text(l)
            tmpKeys = self.r.get_ranked_phrases()
            t, filteredKey = k.preprocess(tmpKeys, i, l)
            codeKeyIndex += t
            codeKeys += filteredKey
        return codeKeys, codeKeyIndex

    def translateBatch(self, codeList):
        inp = np.zeros((len(codeList), 100))
        for i, code in enumerate(codeList):
            for j, w in enumerate(code.split(' ')):
                if j == 100:
                    break
                if w not in self.datstok.w2i.keys():
                    inp[i][j] = 0
                else:
                    inp[i][j] = self.datstok.w2i[w]
        coms = np.zeros((len(codeList), comlen))
        coms[:, 0] = self.st
        
        for i in range(1, comlen):
            if not self.checkMode(self.mode, 'withSbt'):
                results = self.model.predict([inp, coms], batch_size=len(codeList))
            else:
                sml = np.zeros((len(codeList), 100))
                sml[:, :] = np.array(self.sml)
                results = self.model.predict([inp, coms, sml], batch_size=len(codeList))
            for c, s in enumerate(results):
                coms[c][i] = np.argmax(s)
        strComs = [i.split(' ') for i in seqList2Sent(coms, self.comstok)]
        transferedComs = []
        for com in strComs:
            s = ''
            for j in range(1, len(com)):
                if com[j] == '</s>':
                    break
                s += com[j] + ' '
            transferedComs.append(s)
        return [i.split(' ')[: -1] for i in transferedComs]       

    def translateStrs(self, code, sbt=False, sml=None):
        com = ''
        inp = ' '.join(code)
        c = self.translate(inp, sbt, sml)
        for i in range(1, len(c)):
            if c[i] == '</s>':
                break
            com += c[i] + ' '
        return com, c

    def translate(self, code, sbt, sml):
        words = code.split(' ')
        inp = [np.zeros(100)]
        for i, w in enumerate(words):
            if i >= 100:
                break
            if w not in self.datstok.w2i.keys():
                inp[0][i] = 0
            else:
                inp[0][i] = self.datstok.w2i[w]
        coms = np.zeros(comlen)
        coms[0] = self.st
        coms = [coms]

        for i in range(1, comlen):
            if not sbt:
                results = self.model.predict([inp, coms], batch_size=1)
            else:
                results = self.model.predict([inp, coms, sml], batch_size=1)
            for c, s in enumerate(results):
                coms[c][i] = np.argmax(s)
        return seq2sent(coms[0], self.comstok).split(' ')

    def tokenizeCodes(self, _rawCodes, _mode):
        tokenizer = Token()
        if self.checkMode(_mode, 'withToken'):
            tokenedCodes = _rawCodes
        linedCodes = _rawCodes
        if self.checkMode(_mode, 'needLine'):
            linedCodes = tokenizer.generateLinesFromRawCodes(_rawCodes)
        tokenedCodes = tokenizer.getFromLines(linedCodes)
        return tokenedCodes
            
    def loadData(self, _codeFile, _mode):
        # load raw data from different sources
        if self.checkMode(_mode, 'fromFile'):
            code, num = self.loadFromFile(_codeFile, _mode)
        else:
            code, num = self.loadFromString(_codeFile, _mode)
        return code, num

    def loadFromFile(self, _codeFile, _mode):
        if self.checkMode(_mode, 'withLine'):
            code, num = self.loadFromLinedFile(_codeFile, _mode)
        else:
            # 无分行的文件为数据集
            code, num = self.loadDataset(_codeFile, _mode)
        return code, num

    def loadDataset(self, _codeFile, _mode):
        codes = []
        with open(_codeFile) as f:
            for line in f.readlines():
                codes.append([line])
        return codes, len(codes)

    def loadFromString(self, _codeStr, _mode):
        if self.checkMode(_mode, 'withLine'):
            code = _codeStr.split('\n')
        else:
            code = _codeStr
        return [code], 1
            
    def loadFromLinedFile(self, _codeFile, _mode):
        code = []
        with open(_codeFile) as f:
            for line in f.readlines():
                code.append(line)
        return [code], 1

    def loadModel(self, _mode):
        keras.backend.clear_session()
        if self.checkMode(_mode, 'withSbt'):
            model = keras.models.load_model(sbtModelFile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
        else:
            model = keras.models.load_model(modelFile, custom_objects={'top2': top2, 'top3': top3, 'top5':top5})
        return model

    @staticmethod
    def genMode(_withLines, _withToken, _withSbt, _fromFile, _needLine):
        mode = 0
        if _needLine:
            mode = mode * 10 + 1
        if _withLines:
            mode = mode * 10 + 1
        if _withToken:
            mode = mode * 10 + 1
        if _withSbt:
            mode = mode * 10 + 1
        if _fromFile:
            mode = mode * 10 + 1
        return mode
    
    @staticmethod
    def checkMode(_mode, _str):
        if _str == 'needLine':
            return (_mode // 10000) % 10 == 1
        if _str == 'withLine':
            return (_mode // 1000) % 10 == 1
        if _str == 'withToken':
            return (_mode // 100) % 10 == 1
        if _str == 'withSbt':
            return (_mode // 10) % 10 == 1
        if _str == 'fromFile':
            return (_mode) % 10 == 1



if __name__ == '__main__':
    # exp = Explain('../test1.java', 1001)
    code = "public void disconnect() {\ntry {\nsocket.flush();\nsocket.close();\nconnected = false;\nnotifyDisconnect();\n} catch (IOException ex) {\nex.printStackTrace(); } }"
    # exp = Explain('./smtestFile', 10001)
    print(code)
    exp = Explain(_codeFile=None, _mode=1000, model=None)
    exp.reloadData(code, 1000)
    res = exp.explain()
    with open('test.json', 'w') as f:
        json.dump(res, f, indent=2)
