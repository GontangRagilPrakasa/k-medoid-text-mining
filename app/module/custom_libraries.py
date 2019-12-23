from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Rumus Library
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np
import numpy.linalg as LA
import string

stemmer = StemmerFactory().create_stemmer()  # Object stemmer
remover = StopWordRemoverFactory().create_stop_word_remover()  # objek stopword


class Engine:
    def __init__(self):
        self.cosine_score = []
        self.train_set = []  # Documents
        self.test_set = []  # Query

    def addDocument(self, word):
        self.train_set.append(word)

    def setQuery(self, word):
        self.test_set.append(word)

    def process_score(self):
        stopWords = stopwords.words('english')
        vectorizer = CountVectorizer()

        transformer = TfidfTransformer()

        trainVectorizerArray = vectorizer.fit_transform(self.train_set).toarray()
        testVectorizerArray = vectorizer.transform(self.test_set).toarray()

        cx = lambda a, b: round(np.inner(a, b) / (LA.norm(a) * LA.norm(b)), 3)
        #         print testVectorizerArray
        output = []
        for i in range(0, len(testVectorizerArray)):
            output.append([])

        for vector in trainVectorizerArray:
            # print vector
            u = 0
            for testV in testVectorizerArray:
                # print testV
                cosine = cx(vector, testV)
                #                 self.cosine_score.append(cosine)
                #                 bulatin = (round(cosine),2)
                output[u].append((cosine))
                u = u + 1
        return output
        # return testVectorizerArray


def stemmerEN(text):
    porter = PorterStemmer()
    stop = set(stopwords.words('english'))
    text = text.lower()
    text = [i for i in text.lower().split() if i not in stop]
    text = ' '.join(text)
    preprocessed_text = text.translate(None, string.punctuation)
    text_stem = porter.stem(preprocessed_text)
    return text_stem


def preprocess(text):
    text = text.lower()
    text_clean = remover.remove(text)
    text_stem = stemmer.stem(text_clean)
    return text_stem


def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs, cs = np.where(D == 0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    wikwik = np.random.shuffle(index_shuf)
    #     print (wikwik)
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r, c in zip(rs, cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])
    print(M)

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:, M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]

    # return results
    return M, C
