import collections
import itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.metrics import precision, recall, f_measure

def word_feats(words):
	return dict([(word, True) for word in words])

negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]

print('train on %d instances, test on %d instances - Naive Bayes' % (len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
refsets = collections.defaultdict(set)
testsets = collections.defaultdict(set)
 
for i, (feats, label) in enumerate(testfeats):
	refsets[label].add(i)
	observed = classifier.classify(feats)
	testsets[observed].add(i)
 
accuracy = nltk.classify.util.accuracy(classifier, testfeats)
pos_precision = precision(refsets['pos'], testsets['pos'])
pos_recall = recall(refsets['pos'], testsets['pos'])
pos_fmeasure = f_measure(refsets['pos'], testsets['pos'])
neg_precision = precision(refsets['neg'], testsets['neg'])
neg_recall = recall(refsets['neg'], testsets['neg'])
neg_fmeasure =  f_measure(refsets['neg'], testsets['neg'])
		
print('')
print ('---------------------------------------')
print ('             NAIVE BAYES               ')
print ('---------------------------------------')
print ('accuracy:', accuracy)
print ('precision', (pos_precision + neg_precision) / 2)
print ('recall', (pos_recall + neg_recall) / 2)
print ('f-measure', (pos_fmeasure + neg_fmeasure) / 2)