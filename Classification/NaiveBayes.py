from nltk import FreqDist

class NaiveBayes:
    def __init__(self, neg, pos, txt, sentence):
        #allTermNeg berisi semua kata pada kelas negatif dg jmlh kemunculannya
        with open(neg, "r") as negatif :
            self.tweetsNeg = []
            for line in negatif:
                self.tweetsNeg.append(line)
        self.allTermNeg = [] 
        for i in range(len(self.tweetsNeg)):
            self.freqNeg = FreqDist(self.tweetsNeg[i].split())
            self.termNeg = self.freqNeg.most_common()
            self.allTermNeg = self.allTermNeg + [self.termNeg]
        self.allDocNeg = len(self.tweetsNeg) #byk dokumen yg kelasnya negatif
        
         #allTermPos berisi semua kata pada kelas positif dg jmlh kemunculannya
        with open(pos, "r") as positif :
            self.tweetsPos = []
            for line in positif:
                self.tweetsPos.append(line)
        self.allTermPos = [] 
        for i in range(len(self.tweetsPos)):
            self.freqPos = FreqDist(self.tweetsPos[i].split())
            self.termPos = self.freqPos.most_common()
            self.allTermPos = self.allTermPos + [self.termPos]
        self.allDocPos = len(self.tweetsPos) #byk dokumen yg kelasnya positif
        
        self.allDoc = self.allDocNeg +self. allDocPos #byknya seluruh dokumen
        self.pClassNeg = self.allDocNeg / self.allDoc
        self.pClassPos = self.allDocPos / self.allDoc
        
        #menggabung seluruh kata dalam kelas negatif maupun positif
        with open(neg, "r") as negatif :
            self.allNeg = " ".join(line.strip() for line in negatif)
        with open(pos, "r") as positif :
            self.allPos = " ".join(line.strip() for line in positif)
        self.totalTermNeg = len(self.allNeg.split())
        self.totalTermPos = len(self.allPos.split())
        self.totalAllTerm = self.totalTermNeg + self.totalTermPos
        
        if txt != "":
            self.classifyAll(txt)
        else:
            self.classify(sentence)
            
    def classify(self, sentence):
        self.wordTest = sentence.split()
        self.pXNeg = 1
        self.pXPos = 1
        for i in range(len(self.wordTest)):
            for nTermNeg in range(self.allDocNeg):
                if self.wordTest[i] in self.tweetsNeg[nTermNeg]:
                    #pXNeg = pXNeg * 2 / (totalTermNeg + totalAllTerm) 
                    self.pXNeg = self.pXNeg * 2 / (self.totalTermNeg + self.totalAllTerm) * 8000
                else:
                    #pXNeg = pXNeg / (totalTermNeg + totalAllTerm) 
                    self.pXNeg = self.pXNeg / (self.totalTermNeg + self.totalAllTerm) * 8000
            for nTermPos in range(self.allDocPos):
                if self.wordTest[i] in self.tweetsPos[nTermPos]:
                    #pXPos = pXPos * 2 / (totalTermPos + totalAllTerm)
                    self.pXPos = self.pXPos * 2 / (self.totalTermPos + self.totalAllTerm) * 8000
                else:
                    #pXPos = pXPos / (totalTermPos + totalAllTerm)
                    self.pXPos = self.pXPos / (self.totalTermPos + self.totalAllTerm) * 8000
        
        self.pNeg = self.pXNeg * self.pClassNeg
        self.pPos = self.pXPos * self.pClassPos
        
        if self.pPos > self.pNeg:
            return "Positif"
        #else:
        #    return "Negatif"
        elif  self.pPos < self.pNeg:
            return "Negatif"
        else:
            return "Positif/Negatif"
        
    def classifyAll(self, txt):
        file = open("kelas.txt", "w")
        with open(txt, "r") as f:
            tweets = []
            for line in f:
                tweets.append(line)
        TP = 0
        TN = 0 
        for i in range(len(tweets)):
            temp = self.classify(tweets[i])
            file.write(temp + '\n')
            if temp == 'Positif' and i > 41:
                TP = TP + 1
            elif temp == 'Negatif' and i < 42:
                TN = TN + 1
        file.close
        self.akurasi = (TP+TN)/len(tweets)
        return self.akurasi
    
    def classifyTest(self, txt):
        file = open("classTest.txt", "w")
        with open(txt, "r") as f:
            tweets = []
            for line in f:
                tweets.append(line)
        pos = 0
        neg = 0
        net = 0
        for i in range(len(tweets)):
            temp = self.classify(tweets[i])
            file.write(temp + '\n')
            if temp == 'Positif':
                pos = pos + 1
            elif temp == 'Negatif':
                neg = neg + 1
            else:
                net = net + 1
        file.close
        self.result = [pos/len(tweets)*100, neg/len(tweets)*100, net/len(tweets)*100]
        return self.result