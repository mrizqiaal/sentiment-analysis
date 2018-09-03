# -*- coding: utf-8 -*-
import timeit as exe
from Classification import KNN, NaiveBayes

neg = "D:/STUDY/Semester 6/Information Retrieval/Project/6. Data/negfix.txt"
pos = "D:/STUDY/Semester 6/Information Retrieval/Project/6. Data/posfix.txt"
txt = "D:/STUDY/Semester 6/Information Retrieval/Project/6. Data/opinifix2.txt"
sentence = "awesome love"

start = exe.default_timer()
#knn = KNN.KNN(neg, pos, txt, sentence, 1)
#print(knn.classifyAll(txt, 1))
nb = NaiveBayes.NaiveBayes(neg, pos, txt, sentence)
print(nb.classifyAll(txt))
end = exe.default_timer()
exeTime = end - start
print(exeTime)
