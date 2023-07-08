import re
import pathlib
import string
import math
import json
from nltk.stem import WordNetLemmatizer

def getStopWords():
    """function to extract stopwords from the file"""
    stopWordsList = []
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        stop = open(path + "\\Stopword-List.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        for stopWord in stop:
            for word in stopWord.strip().split("\n"):
                if word not in string.whitespace:
                    stopWordsList.append(word)          #getting stop words from file and storing them in a list
        stop.close()

    return stopWordsList

def getDocuments():                                     
    """function to extract data from documents"""
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    rawDataList = []                                    #to store data from documents
    
    for fileNum in range(1, 51):                        #getting data from the dataset
        try:
            data = open(path + "\\ShortStories\\" + str(fileNum) + ".txt", "r", encoding = "utf-8")
        except IOError:
            print("Error: File not found on the specified path.")
        else:
            readData = data.read()
            readData = re.sub(r"don’t", "do not", readData)         #cleaning data
            readData = re.sub(r"doesn’t", "does not", readData)
            readData = re.sub(r"won’t", "will not", readData)
            readData = re.sub(r"can’t", "can not", readData)
            readData = re.sub(r"shan’t", "shall not", readData)
            readData = re.sub(r"haven’t", "have not", readData)
            readData = re.sub(r"I’ll", "I will", readData)
            readData = re.sub(r"you’ll", "you will", readData)
            readData = re.sub(r"’s", " ", readData)
            rawDataList.append(readData)                            #store data after cleaning
            data.close()
        
    return rawDataList

def Preprocessing():
    """function to preprocess data (clean data, casefold, remove stopwords, lemmatization)"""

    rawDataList = []
    stopWordsList = []
    PreprocessedData = dict()
    lemmatizer = WordNetLemmatizer()       #create object of class WordNetLemmatizer from nltk library

    stopWordsList = getStopWords()

    rawDataList = getDocuments()

    for i in range(len(rawDataList)):
        tempList = []
        tempList = rawDataList[i]
        # tempList = re.sub(r"[-]" , "", tempList)
        tempList = re.sub(r"[\n,?'-.—;:!“”’()#/@{}|<>`_+=~\"\\]" , " ", tempList)       #clean data

        breakList = []                      #list for storing data word by word
        tempList = tempList.lower()         #casefolding
        breakList = tempList.split(" ")     #split the data
        WordsList = []                      #list for storing preprocessed data
        for word in breakList:
            word = word.strip()
            if word not in string.whitespace and word not in stopWordsList:
                word = lemmatizer.lemmatize(word)   #lemmatization
                WordsList.append(word)
        index = i + 1                       #index should correspond to Document Number
        PreprocessedData[index] = WordsList
        
    return PreprocessedData

def CreateInvertedIndex(PreprocessedData):
    """function to create inverted index"""
    InvertedIndex = {}
    for key in PreprocessedData:
        WordsList = []                                  #list for storing one document data
        WordsList = PreprocessedData[key]
        for term in WordsList:
            if term in InvertedIndex:
                if key not in InvertedIndex[term][1]:   #if document number not present in inverted index, increment document frequency
                    InvertedIndex[term][0] += 1
            else:
                InvertedIndex[term] = []                #if term not present in inverted index, add it
                InvertedIndex[term].append(1)           #store document frequency
                InvertedIndex[term].append([])          #create a list for storing document numbers
            
            if key not in InvertedIndex[term][1]:
                InvertedIndex[term][1].append(key)      #if document number not present in inverted index, add it


    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\InvertedIndex.txt", "w", encoding = "utf-8")       #store inverted index in a file
    except IOError:
        print("Error: Operation failed.")
    else:
        filePtr.write(json.dumps(InvertedIndex))
        filePtr.close()

    return InvertedIndex

def ComputeTermFrequency(PreprocessedData):             
    """function to compute term frequency"""
    termFrequency = dict()                              #dictionary to store term frequency

    for key in PreprocessedData:
        WordsList = []                                  #list for storing one document data
        WordsList = PreprocessedData[key]               #get data of one document
        termFrequency[key] = {}                         #initialize another dictionary at document number

        for term in WordsList:
            
            if term not in termFrequency[key]:          #if the term freuqency has been calculated earlier, don't recalculate. Move on to next term.
                countOccurence = 0
                for word in WordsList:
                    if term == word:
                        countOccurence += 1
                
                termFrequency[key][term] = countOccurence       #store the term frequency
    
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\TermFrequency.txt", "w", encoding = "utf-8")       #store term frequency in a file
    except IOError:
        print("Error: Operation failed.")
    else:
        filePtr.write(json.dumps(termFrequency))
        filePtr.close()

    return termFrequency

def Compute_idf(N):
    """function to calculate idf"""

    idf = dict()                                    #dictionary to store idf of terms

    InvertedIndex = LoadInvertedIndex()             #load inverted index
    
    for term in InvertedIndex:
        df = InvertedIndex[term][0]                 #get document frequency
        """Formula: idf = log(df)/N"""
        numerator = math.log10(df)
        fraction = numerator/N
        idf_val = fraction
        idf[term] = idf_val                         #store idf of the term in the dictionary after calculation
    
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\InverseDocumentFrequency.txt", "w", encoding = "utf-8")       #store idf in a file
    except IOError:
        print("Error: Operation failed.")
    else:
        filePtr.write(json.dumps(idf))
        filePtr.close()

    return idf

def Compute__tf_idf():
    """function to calculate tf-idf weights"""

    tempDict = dict()                                       #dictionary to store term frequencies of terms of one document                                       
    tf_idf = dict()                                         #dictionary to store tf-idf weights                                

    TermFrequency = LoadTermFrequency()                     #load Term Frequency from file
    idf = LoadInverseDocumentFrequency()                    #load Inverse Document Frequency from file

    for docNum in TermFrequency:
        tempDict = TermFrequency[docNum]                    #store term frequencies of terms of one document
        tf_idf[docNum] = {}                                 #initialize another dictionary at key = document number to store tf-idf weights of the document at its corresponding index
        for term in tempDict:
            tf = TermFrequency[docNum][term]                #get the term frequency
            tf_idf[docNum][term] = tf * idf[term]           #calcualte tf*idf

    path = pathlib.Path(__file__).parent.absolute()                         #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\tf-idf.txt", "w", encoding = "utf-8")       #store tf-idf weights in a file
    except IOError:
        print("Error: Operation failed.")
    else:
        filePtr.write(json.dumps(tf_idf))
        filePtr.close()  

    return tf_idf

def docMagnitudes():
    """function to calculate magnitudes of tf-idf weights documents"""

    docMag = dict()                                                         #dictionary to store document tf-idf weights magnitude of each document

    tf_idf = Load__tf_idf()                                                 #load tf-idf weights from file

    for docNum in tf_idf:
        tempDict = tf_idf[docNum]                                           #get tf-idf weights for one document for computation
        Magnitude = 0                                                       #initialize magnitude
        for term in tempDict:
            tf_idf_val = tempDict[term]
            Magnitude += (tf_idf_val ** 2)                                   #calculating magnitude

        docMag[docNum] = math.sqrt(Magnitude)                                #store magnitude at key = document number

    return docMag

def ProcessQuery(query):
    """function for query processing"""
    words = []                                                                      #list to store terms
    lemmatizer = WordNetLemmatizer()                                                #create object of class WordNetLemmatizer from nltk library
    stopWordsList = getStopWords()
    queryList = query.split(" ")                                                    #store the entire query word by word

    InvertedIndex = LoadInvertedIndex()                                             #load data from file

    for word in queryList:
        word = word.lower()                                                         #casefolding
        # word = re.sub(r"[-]" , "", word)
        word = re.sub(r"[\n,?'-.—;:!“”’()#/@{}|<>`_+=~\"\\]" , " ", word)           #cleaning query
        if word not in stopWordsList and word not in string.whitespace:
            word = lemmatizer.lemmatize(word)                                       #lemmatization
            words.append(word)

    tfQuery = dict()                                                                #dictionary to store tf values of terms in query

    for term in words:
        if term not in tfQuery:                                                     #if the term freuqency has been calculated earlier, don't recalculate. Move on to next term.
            countOccurence = 0
            for word in words:
                if term == word:
                    countOccurence += 1
                    
            tfQuery[term] = countOccurence                                          #store term frequency of query
    
    return tfQuery
    
def computeSimilarity(tfQuery, docMag):
    """function to calculate Cosine Similarity"""

    tf_idf_Query = dict()                   #dictionary to store tf-idf weight of query
    Magnitude_Query = 0                     #variable to store magnitude of query
    Magnitude = 0                           #temporary variable used in magnitude calculation
    Similarity = dict()                     #dictionary to store cosine similarity of query with every document
    denominator = 0                         #variable to store denominator of the formula of cosine similarity

    idf = LoadInverseDocumentFrequency()
    tf_idf_doc = Load__tf_idf()

    for term in tfQuery:
        if term in idf:                                         #if query term exists in dataset
            tf_idf_Query[term] = tfQuery[term] * idf[term]      #calculating tf-idf weights of query
    
    for term in tf_idf_Query:                                   #calculating magnitude of query tf-idf weights
        tf_idf_val = tf_idf_Query[term]
        Magnitude += (tf_idf_val ** 2)
        Magnitude_Query = math.sqrt(Magnitude)

    for docNum in tf_idf_doc:
        
        tf_idf = tf_idf_doc[docNum]                            #get tf-idf weights of a document for computation
        numerator = 0                                          #initializing numerator of cosine similarity formula

        for term in tf_idf_Query:
            if term in tf_idf:                                #if query term exists in a document
                product = tf_idf[term] * tf_idf_Query[term]   #calculating di*qi
                numerator += product                          #summation of values

        denominator = docMag[docNum] * Magnitude_Query        #calculating denominator of cosine similarity formula
        
        try:                                                 #to prevent the program from crashing due to Divide by Zero error
            Similarity[docNum] = numerator / denominator
        except ZeroDivisionError:
            Similarity = 0
        else:
            pass
              
    return Similarity

def Results(Similarity, ranked):
    """function to filter results and rank documents if the ranked parameter is set to True"""
    Alpha = 0.005
    filtered_result = {}                                   #dictionary to store result after filteration
    result_docs = []                                       #dictionary to store final result

    print(Similarity)

    for docNum in Similarity:
        if Similarity[docNum] >= Alpha:
            filtered_result[docNum] = Similarity[docNum]
    
    if not filtered_result:                                 #if no result found then return 0
        return 0
    
    if ranked == True:                                      #if this parameter is true then rank documents
        result = {key: val for key, val in sorted(filtered_result.items(), key=lambda item: item[1], reverse=True)}     #rank documents according to descending order of the Cosine Similarity value
    else:
        result = filtered_result                            #if rank = False just store the filtered result

    for key in result:
        result_docs.append(key)                             #store document numbers as the final result

    return result_docs

def LoadInvertedIndex():
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\InvertedIndex.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        InvertedIndex = json.load(filePtr)             #load inverted index
        filePtr.close()
    
    return InvertedIndex

def LoadTermFrequency():
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\TermFrequency.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        TermFrequency = json.load(filePtr)             #load term frequency
    
    return TermFrequency

def LoadInverseDocumentFrequency():
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\InverseDocumentFrequency.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        InverseDocumentFrequency = json.load(filePtr)             #load Inverse Document Frequency
        filePtr.close()

    return InverseDocumentFrequency

def Load__tf_idf():
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\Files_Load_And_Store\\tf-idf.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        tf_idf = json.load(filePtr)             #load Inverse Document Frequency
        filePtr.close()

    return tf_idf

# def getQuery():
#     query = 'sent due'
#     return query

print("Loading....")  

# PreprocessedData = Preprocessing()
# InvertedIndex = CreateInvertedIndex(PreprocessedData)
# TermFrequency = ComputeTermFrequency(PreprocessedData)
# N = len(PreprocessedData)
# idf = Compute_idf(N)
# tf_idf = Compute__tf_idf()
# docMag = docMagnitudes()
# query = getQuery()
# tfQuery = ProcessQuery(query)
# Similarity = computeSimilarity(tfQuery, docMag)
# result_docs = Ranking(Similarity, ranked = True)
# print(result_docs)