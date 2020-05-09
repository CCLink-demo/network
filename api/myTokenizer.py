import re
import wordninja
operator = ['+','-','*','/','%','++','--','+=','-=','+=','/=',
            '==','!=','>','<','>=','<=',
            '&','|','^','~','<<','>>','>>>',
            '&&','||','!',
            '=','+=','-=','*=','/=','%=','<<=','>>=','&=','^=','|=',
            '?:']

delimiters = ['{','}','[',']','(',')','.',',',':',';']

stop = ['\'']

reserved = ['abstract','assert','boolean','break','byte',
            'case','catch','char','class','const',
            'continue','default','do','double','else',
            'enum','extends','final','finally','float',
            'for','goto','if','implements','import',
            'instanceof','int','interface','long','native',
            'new','package','private','protected','public',
            'return','short','static','strictfp','super',
            'switch','synchronized','this','throw','throws',
            'transient','try','void','volatile','while','args','throws']

class Token(object):
 
    def __init__(self):
        self.results = []
 
        self.lineno = 1
 
        self.keywords = ['abstract','assert','boolean','break','byte',
            'case','catch','char','class','const',
            'continue','default','do','double','else',
            'enum','extends','final','finally','float',
            'for','goto','if','implements','import',
            'instanceof','int','interface','long','native',
            'new','package','private','protected','public',
            'return','short','static','strictfp','super',
            'switch','synchronized','this','throw','throws',
            'transient','try','void','volatile','while','args','throws']
        Keyword = r'(?P<Keyword>(abstract){1}' 
        for kw in self.keywords:
            Keyword += '|({0}){{1}}'.format(kw)
        Keyword += ')'

        Operator = r'(?P<Operator>\+\+|\+=|\+|--|-=|-|\*=|/=|/|%=|%|=|<|<=|>|>=)'

        Separator = r'(?P<Separator>[,:\{}:)(<>])'
        
        Number = r'(?P<Number>\d+[.]?\d+|0)'
 
        ID = r'(?P<ID>[a-zA-Z_][a-zA-Z_0-9]*)'
 
        Method = r'(?P<Method>[a-zA-Z_][a-zA-Z_0-9]*\(.*\))'
 
        Error = r'\"(?P<Error>.*)\"'
 
        self.patterns = re.compile('|'.join([Keyword, Method, ID, Number, Separator, Operator, Error]))
 
    def read_file(self, filename):
         with open(filename, "r") as f_input:
               return [line.strip() for line in f_input]
 
    def write_file(self, lines, filename = 'D:/results.txt'):
        with open(filename, "a") as f_output:
                for line in lines:
                    if line:
                        f_output.write(line)
                    else:
                        continue
 
    def get_token(self, line):
 
        for match in re.finditer(self.patterns, line):
 
            yield (match.lastgroup, match.group())
 
    def run(self, line, flag=True):
        words = []
        for token in self.get_token(line):
            if flag:
                print ("line %3d :" % self.lineno, token)
            if token[0] in {'Keyword', 'ID', 'Method', 'Error', 'Number'}:
                words.append(token)
        return words
    
    def get(self, code):
        lines = code.split('\n')
        tokenizedLine = []
        for line in lines:
            tmp = []            
            words = self.run(line, False)
            if words == []:
                continue
            for w in words:
                if w[0] in ['ID', 'Error', 'Method']:
                    tmp += wordninja.split(w[1].lower())
                else:
                    tmp.append(w[1],lower())
            tokenizedLine.append(tmp)
        print(tokenizedLine)
        return tokenizedLine
    
    def generateLinesFromRawCodes(self, rawCodes):
        retLines = []
        for code in rawCodes:
            tmpLines = []
            for l in re.split('[{;}]', code[0]):
                if not l.isspace():
                    tmpLines.append(l)
            retLines.append(tmpLines)
        return retLines

    def getLinesFromRawCodes(self, rawCodes):
        retLines = []
        for line in rawCodes:
            tmpLines = []
            for l in re.split('[{;}]', line):
                if not l.isspace():
                    tmpLines.append(l)
            retLines.append(tmpLines)
        return retLines
    
    def getFromLinedCode(self, code):
        tokenizedCode = []
        for line in code:
            tmp = []
            words = self.run(line, False)
            if words == []:
                continue
            for w in words:
                if w[0] in ['ID', 'ERROR', 'Method']:
                    tmp += wordninja.split(w[1].lower())
                else:
                    tmp.append(w[1].lower())
            tokenizedCode.append(tmp)
        tokenizedCode = self.toLineList(tokenizedCode)
        return tokenizedCode
    
    def getFromFrontCode(self, code):
        tokenizedCode = []
        linNums = []
        for i, line in enumerate(code):
            tmp = []
            tmpNum = []
            words = self.run(line, False)
            if words == []:
                continue
            for w in words:
                if w[0] in ['ID', 'ERROR', 'Method']:
                    tmp += wordninja.split(w[1].lower())
                else:
                    tmp.append(w[1].lower())
            tokenizedCode.append(tmp)
            linNums.append(i)
        tokenizedCode = self.toLineList(tokenizedCode)
        return tokenizedCode, linNums
    
    def getFromLines(self, codes):
        tokenizedCodes = []
        for i, code in enumerate(codes):
            tokenized = self.getFromLinedCode(code)
            tokenizedCodes.append(tokenized)
        return tokenizedCodes

    def getFromFile(self, file):
        f = open(file, 'r')
        lines = f.readlines()
        tokenizedLine = []
        for line in lines:
            tmp = []            
            words = self.run(line, False)
            if words == []:
                continue
            # print(words)
            for w in words:
                if w[0] in ['ID', 'Error', 'Method']:
                    tmp += wordninja.split(w[1].lower())
                else:
                    tmp.append(w[1].lower())
            # print(tmp)
            tokenizedLine.append(tmp)
        # print(1, tokenizedLine)
        tokenizedLine = self.toLineList(tokenizedLine)
        # print(2, tokenizedLine)
        return tokenizedLine

    @staticmethod
    def toDoubleList(lineList):
        retList = []
        for l in lineList:
            retList.append(l.split(' '))
        return retList
    
    @staticmethod
    def toLineList(doubleList):
        retList = []
        for line in doubleList:
            retList.append(' '.join(line))
        return retList
 
    def printrun(self, line, flag = True):
        for token in self.get_token(line):
            if flag:
                print ("lines x: ", token)


class Keyword(object):
    def __init__(self):
        super().__init__()
        self.reserved = reserved
    
    def preprocess(self, keyList, lineId, line):
        retList = []
        filteredCode = []
        for key in keyList:
            # find the index of the first word
            index = line.find(key)
            if index < 0:
                continue
            tmp = len(line[: index].split(' '))
            
            tmpList = []
            s = ''
            for i, w in enumerate(key.split(' ')):
                if w in self.reserved:
                    s += '<reserved> '
                    continue
                s += w + ' '
                tmpList.append(i + tmp - 1)
            if len(tmpList) > 0:
                retList.append((lineId, tmpList))
                filteredCode.append(s)
        return retList, filteredCode

 
 
if __name__=='__main__':
    token = Token()
    filepath = "test.java"
 
    code = '''
    public static void main(String[] args) throws ScriptException, NoSuchMethodException {
        NashornScriptEngine engine = (NashornScriptEngine) new ScriptEngineManager().getEngineByName("nashorn");
        engine.eval("load('res/nashorn10.js')");

        long t0 = System.nanoTime();

        for (int i = 0; i < 100000; i++) {
            engine.invokeFunction("testPerf");
        }

        long took = System.nanoTime() - t0;
        System.out.format("Elapsed time: %d ms", TimeUnit.NANOSECONDS.toMillis(took));
    }
    '''
    token.getFromFile('test.java')

