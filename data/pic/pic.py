import sys
import os
from os.path import basename

for i in range(1, 35, 1):
    imgUrl = 'http://image.slidesharecdn.com/20141118deeplearningforsentiment-atsgaico2014-bymarkcieliebak-141119041153-conversion-gate02/95/can-deep-learning-solve-the-sentiment-analysis-problem-' + str(
        i) + '-1024.jpg'
    os.system('wget -nc ' + imgUrl + '.jpg' + ' -O ' + str(i) + '.jpg')
    print('Descargada imagen ' + 'i')
