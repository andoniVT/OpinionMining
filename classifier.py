from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from model import VectorModel as VM 

class SupervisedClassifier(object):
    
    def __init__(self , data=None, labels=None):
        print "Suppervised Classifier"
        self.data = data
        self.labels = labels
        self.classifier = []
        
    def train(self):
        pass
    
    def classify(self, test_data):
        pass 
    
    def set_classifier(self , classifier):
        self.classifier = classifier

class NaiveBayes(SupervisedClassifier):
    
    def __init__(self , data=None , labels = None):
        SupervisedClassifier.__init__(self , data , labels)
        print " -- Naive Bayes Classifier -- "
    
    def train(self):
        super(NaiveBayes , self).train()
        classifier = MultinomialNB()
        classifier = classifier.fit(self.data, self.labels)
        self.classifier = classifier
        return classifier
    
    def classify(self, test_data):
        super(NaiveBayes , self).classify(test_data)
        predictions = []
        for i in test_data:
            value = self.classifier.predict(i)
            predictions.append(value)
        return predictions
    
    
class SVM(SupervisedClassifier):
    
    def __init__(self , data=None , labels = None):
        SupervisedClassifier.__init__(self , data , labels)
        print " -- Support Vector Machine Classifier -- "
    
    def train(self):
        super(SVM , self).train()
        classifier = LinearSVC()
        classifier = classifier.fit(self.data, self.labels)
        self.classifier = classifier
        return classifier
    
    def classify(self, test_data):
        super(SVM , self).classify(test_data)
        predictions = []
        for i in test_data:
            value = self.classifier.predict(i)
            predictions.append(value)
        return predictions
        
class DecisionTree(SupervisedClassifier):
    
    def __init__(self , data=None , labels = None):
        SupervisedClassifier.__init__(self , data , labels)
        print " -- Decision Tree Classifier -- "
    
    def train(self):
        super(DecisionTree , self).train()
        classifier = DecisionTreeClassifier()
        classifier = classifier.fit(self.data, self.labels)
        self.classifier = classifier
        return classifier
    
    def classify(self, test_data):
        super(DecisionTree , self).classify(test_data)
        predictions = []
        for i in test_data:
            value = self.classifier.predict(i)
            predictions.append(value)
        return predictions  

class MaxEnt(SupervisedClassifier):
    
    def __init__(self , data=None , labels = None):
        SupervisedClassifier.__init__(self , data , labels)
        print " -- Maximum Entropy Classifier -- "
    
    def train(self):
        super(MaxEnt , self).train()
        classifier = LogisticRegression()
        classifier = classifier.fit(self.data, self.labels)
        self.classifier = classifier
        return classifier
    
    def classify(self, test_data):
        super(MaxEnt , self).classify(test_data)
        predictions = []
        for i in test_data:
            value = self.classifier.predict(i)
            predictions.append(value)
        return predictions      
            

if __name__ == '__main__':
    
    '''
    data = [[1,2,3],[4,5,6],[3,2,1],[6,5,4],[2,5,8]]
    labels = [1,1,0,0,1]
    
    nb = NaiveBayes(data,labels)
    nb.train()
    test = [[1,2,3],[3,2,1]]
    print nb.classify(test)
    '''

    corpus = ['Todos estamos felices por los resultados logrados en el ultimo examen' , 
                'Me siento muy contento por la familia que tengo' , 
                'La seleccion de futbol cada vez juega peor' , 
                'la politica de nuestro pais es un desastre' , 
                'el dia de ayer la tienda estuvo cerrada']

    # Cada comentario del corpus debe tener su respectiva etiqueta de clase
    labels = ['Positivo' , 'Positivo','Negativo' , 'Negativo' , 'Neutro']

    vecModel = VM(corpus)
    corpus_tf_idf = vecModel.prepare_models()[3]
    for i in corpus_tf_idf:
        print i 

    classifierSVM = SVM(corpus_tf_idf, labels)
    classifierSVM.train()

    






    







    