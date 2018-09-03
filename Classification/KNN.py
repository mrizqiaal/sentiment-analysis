from nltk import FreqDist
import math
from collections import  Counter

class KNN:
    def __init__(self, neg, pos, txt, sentence, k):
        #allTermNeg berisi semua kata pada kelas negatif dg jmlh kemunculannya (tf)
        with open(neg, "r") as negatif:
            self.tweetsNeg = []
            for line in negatif:
                self.tweetsNeg.append(line)
            self.allTermNeg = [] 
            for i in range(len(self.tweetsNeg)):
                self.freqNeg = FreqDist(self.tweetsNeg[i].split())
                self.termNeg = self.freqNeg.most_common()
                self.allTermNeg = self.allTermNeg + [self.termNeg]
        #normalized tf
        for i in range(len(self.tweetsNeg)):
            for j in range(len(self.allTermNeg[i])):
                self.allTermNeg[i][j] = (self.allTermNeg[i][j][0], self.allTermNeg[i][j][1] / len(self.tweetsNeg[i].split()))
        
        #allTermPos berisi semua kata pada kelas positif dg jmlh kemunculannya (tf)  
        with open(pos, "r") as positif:
            self.tweetsPos = []
            for line in positif:
                    self.tweetsPos.append(line)
            self.allTermPos = [] 
            for i in range(len(self.tweetsPos)):
                self.freqPos = FreqDist(self.tweetsPos[i].split())
                self.termPos = self.freqPos.most_common()
                self.allTermPos = self.allTermPos + [self.termPos]
        #normalized tf
        for i in range(len(self.tweetsPos)):
            for j in range(len(self.allTermPos[i])):
                self.allTermPos[i][j] = (self.allTermPos[i][j][0], self.allTermPos[i][j][1] / len(self.tweetsPos[i].split()))
                
        self.tf = self.allTermNeg + self.allTermPos
        self.allDoc = len(self.tf)

    def classify(self, sentence, k):
        self.query = sentence
        #idf
        self.idf = []
        self.query = self.query.split()
        for i in range(len(self.query)):
            #menghitung byknya jml dokumen dengan term tertentu
            self.numDocWithThisTerm = 0
            for j in range(self.allDoc):
                for k in range(len(self.tf[j])):
                    if self.query[i] in self.tf[j][k]:
                        self.numDocWithThisTerm = self.numDocWithThisTerm + 1
            if self.numDocWithThisTerm > 0:
                self.temp =  1.0 + math.log(float(self.allDoc / self.numDocWithThisTerm))
            else:
                self.temp =  1.0
            self.idf = self.idf + [(self.query[i], self.temp)]
            
        #tfidf
        self.tfidf = []
        for i in range(len(self.query)):
            for j in range(self.allDoc):
                #get tf dgn term tertentu
                for k in range(len(self.tf[j])):
                    if self.query[i] in self.tf[j][k]:
                        self.tfTemp = self.tf[j][k][1]
                        break;
                    else:
                        self.tfTemp = 0
                #get idf dgn term tertentu
                for l in range(len(self.idf)):
                    if self.query[i] in self.idf[l]:
                        self.idfTemp = self.idf[l][1]
                self.tfidf = self.tfidf + [self.tfTemp*self.idfTemp]
        self.tfidf = [i for i in zip(*[iter(self.tfidf)]*self.allDoc)]
        
        #tf query
        self.freqQuery = FreqDist(self.query)
        self.tfQuery = self.freqQuery.most_common()
        #normalized tf query
        for i in range(len(self.tfQuery)):
            self.tfQuery[i] = (self.tfQuery[i][0], self.tfQuery[i][1] / len(self.query))
        #idf query = idf (sama)
        #tf idf query
        self.tfidfQuery = []
        for i in range(len(self.tfQuery)):
            self.tfidfQuery = self.tfidfQuery + [self.tfQuery[i][1]*self.idf[i][1]]
        
        #cosine similarity
        self.cosSim = []
        for i in range(self.allDoc):
            self.dotProduct = 0
            self.queryAll = 0
            self.docI = 0
            for j in range(len(self.tfQuery)):
                self.dotProduct = self.dotProduct + (self.tfidfQuery[j] * self.tfidf[j][i])
                self.queryAll = self.queryAll + (self.tfidfQuery[j] ** 2)
                self.docI = self.docI + (self.tfidf[j][i] ** 2)
            if self.docI == 0:
                self.temp = 0 #mengatasi jika term sama sekali tdk muncul, shg hasil tdk akan menjadi tak hingga
            else:
                self.temp = self.dotProduct / (math.sqrt(self.queryAll) * math.sqrt(self.docI))
            self.cosSim = self.cosSim + [self.temp]
            
        self.indexMax = [i for i, val in enumerate(self.cosSim) if val == max(self.cosSim)]
        self.closestClass = []
        for i in range(len(self.indexMax)):
            if self.indexMax[i]>=len(self.allTermNeg):
                self.closestClass = self.closestClass + ['Positif']
            else:
                self.closestClass = self.closestClass + ['Negatif']
    
        self.freqClass = Counter(self.closestClass).most_common()
        if len(self.freqClass) == 2 and self.freqClass[0][1] == self.freqClass[1][1]:
            return 'Positif/Negatif'
        elif k > len(self.closestClass):
            self.closestClass = sorted(self.closestClass, key = self.closestClass.count, reverse = True)
            return self.closestClass[0]
        else:
            #self.closestClass = sorted(self.closestClass, key = self.closestClass.count)
            return self.closestClass[k-1]
        
    def classifyAll(self, txt, k):
        file = open("kelas2.txt", "w")
        with open(txt, "r") as f:
            tweets = []
            for line in f:
                tweets.append(line)
        TP = 0
        TN = 0
        for i in range(len(tweets)):
            temp = str(self.classify(tweets[i], k))
            file.write(temp + '\n')
            if temp == 'Positif' and i > 41:
                TP = TP + 1
            elif temp == 'Negatif' and i < 42:
                TN = TN + 1
        file.close
        self.akurasi = (TP+TN)/len(tweets)
        return self.akurasi
    
    def classifyTest(self, txt, k):
        file = open("classTest2.txt", "w")
        with open(txt, "r") as f:
            tweets = []
            for line in f:
                tweets.append(line)
        pos = 0
        neg = 0
        net = 0
        for i in range(len(tweets)):
            temp = self.classify(tweets[i], k)
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