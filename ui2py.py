# -*- coding: utf-8 -*-

from PyQt5 import uic

name = "gui"

fin = open(name + ".ui", 'r')
fout = open(name + ".py", 'w+')
uic.compileUi(fin, fout, execute = False)
fin.close()
fout.close()