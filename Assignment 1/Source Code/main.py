import re
import pathlib
import string
import json
from nltk.stem import PorterStemmer

def getStopWords():                                     #function to extract stopwords from the file
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

def getDocuments():                                     #function to extract data fom documents
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

def Preprocessing():                #function to preprocess data (clean data, casefold, remove stopwords, stemming)

    rawDataList = []
    stopWordsList = []
    PreprocessedData = dict()
    stemmer = PorterStemmer()       #create object of class PorterStemmer from nltk library

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
                word = stemmer.stem(word)   #stemming
                WordsList.append(word)
        index = i + 1                       #index should correspond to Document Number
        PreprocessedData[index] = WordsList
        
    return PreprocessedData

def CreateInvertedIndex(PreprocessedData):              #function to create inverted index
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
        filePtr = open(path + "\\InvertedIndex.txt", "w", encoding = "utf-8")       #store inverted index in a file
    except IOError:
        print("Error: Operation failed.")
    else:
        filePtr.write(json.dumps(InvertedIndex))
        filePtr.close()

    return InvertedIndex

def CreatePositionalIndex():        #function to create positional index
    stemmer = PorterStemmer()       #create object of class PorterStemmer from nltk library
    rawDataList = []                #to store data from documents
    stopWordsList = []              
    documents = dict()              #to store document data
    PositionalIndex = dict()        
    tempDict = dict()               #for processing purposes

    stopWordsList = getStopWords()

    rawDataList = getDocuments()
   
    """Preprocessing for positional index done separately because I did not filter stop words while calculating indexes, I filtered them while storing terms in the positional index dictionary"""

    for i in range(len(rawDataList)):   #preprocess data 
        tempList = []
        tempList = rawDataList[i]
        # tempList = re.sub(r"[-]" , "", tempList)
        tempList = re.sub(r"[\n,?'-.—;:!“”’()#/@{}|<>`_+=~\"\\]" , " ", tempList)       #clean data

        breakList = []                                                                  #list for storing data word by word
        tempList = tempList.lower()                                                     #casefolding
        breakList = tempList.split(" ")                                                 #split the data
        WordsList = []                                                                  #list for storing preprocessed data
        for word in breakList:
            word = word.strip()
            if word not in string.whitespace:
                word = stemmer.stem(word)                                                #stemming
                WordsList.append(word)
        index = i + 1                                                                    #index should correspond to Document Number
        documents[index] = WordsList

    for docNum in documents:                                                             #loop where processing for positional index starts
        WordsList = []                                                                   #list for storing data from documents
        tempList = []                                                                    #list made for copying WordsList for comparisions
        WordsList = documents[docNum]
        tempList = WordsList

        for term in WordsList:
            if term not in stopWordsList:
                if term not in tempDict:                                                  #to avoid duplication
                    tempDict[term] = []
                elif term in tempDict and docNum in tempDict[term]:
                    continue

                countOccurence = 0
                index = 0
                posList = []                                                              #list for storing positions of a term

                for word in tempList:
                    index += 1                                                            #increment index for each word
                    if term == word:                                                      #if the current term is found in the document increment occurence
                        countOccurence += 1
                        posList.append(index)                                             #store the position of the word in the document
                        tempDict[term].append(docNum)                                     #to avoid duplication, store it for the duplication check
                    else:
                        pass

                if term not in PositionalIndex:                                           #create positional index
                    PositionalIndex[term] = []
                    PositionalIndex[term].append(countOccurence)
                    PositionalIndex[term].append({})
                    PositionalIndex[term][1][docNum] = posList

                else:
                    if docNum not in PositionalIndex[term][1]:
                        PositionalIndex[term][1][docNum] = posList
                    else:
                        tempPosList = PositionalIndex[term][1][docNum]                   #update positional index
                        tempPosList.append(posList)
                        PositionalIndex[term][1][docNum] = tempPosList

                    PositionalIndex[term][0] += countOccurence                          #store occurence 
    
    path = pathlib.Path(__file__).parent.absolute()                                     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\PositionalIndex.txt", "w", encoding = "utf-8")         #store positional index in a file
    except IOError:
        print("Error: Operation failed.")
    else:
        filePtr.write(json.dumps(PositionalIndex))
        filePtr.close()

    return PositionalIndex

def AndOrOperation(operator, docList1, docList2, result):                             #function to run And/Or Operations 

    if operator == "and":
        docList = [docNum for docNum in docList1 if docNum in docList2]
        result = docList
    
    if operator == "or":
        result = list(set(result + docList1))
        result = list(set(result + docList2))
        result = sorted(result)
    
    return result


def ProcessQuery(query):                                                            #function for query processing
    operators = []                                                                  #list to store operators
    words = []                                                                      #list to store terms
    result = []
    stemmer = PorterStemmer()                                                       #create object of class PorterStemmer from nltk library
    stopWordsList = getStopWords()
    queryList = query.split(" ")                                                    #store the entire query word by word

    InvertedIndex = LoadInvertedIndex()                                             #load data from file

    for word in queryList:
        word = word.lower()                                                         #casefolding
        # word = re.sub(r"[-]" , "", word)
        word = re.sub(r"[\n,?'-.—;:!“”’()#/@{}|<>`_+=~\"\\]" , " ", word)           #cleaning query
        if word == "and" or word == "or" or word == "not" and word not in string.whitespace:
            operators.append(word)
        else:
            if word not in stopWordsList:
                word = stemmer.stem(word)                                           #stemming
                words.append(word)

    if len(words) == 1 and not operators:                                           #if its a single word
        term1 = words.pop(0)
        try:
            result = InvertedIndex[term1][1]
        except KeyError:
            return 0
        else:
            return result

    if operators and not words:                                                    #if no terms entered, only operators
        result = None
        return result

    if (queryList[0].lower()) == "not":                                             #set flag if the first word is a not operator
        notFlag = True
    else:
        notFlag = False

    i = 0                                                                           #to count iterations

    while operators:

        i += 1

        operator = operators.pop(0)
        
        if not result and notFlag != True and i == 1:                               #if its the first iteration and first word is not the not operator
            try:
                term1 = words.pop(0)
                docList1 = InvertedIndex[term1][1]
                term2 = words.pop(0)
                docList2 = InvertedIndex[term2][1]
            except KeyError:
                return 0
            else:
                pass

        elif notFlag == True:                                                   #if the first word is the not operator, operation on only one word is required
            term1 = words.pop(0)
            try:
                docList1 = InvertedIndex[term1][1]
            except KeyError:
                return 0
            else:
                notFlag = False
        else:                                                                   #if previously part of query is processed, get the new word and the previous result
            term1 = words.pop(0)
            try:
                docList1 = InvertedIndex[term1][1]
            except KeyError:
                return 0
            else:
                docList2 = result

        if operators and (operator == "and" or operator == "or") and operators[0] == "not":     #if the not operator occurs with and/or
            operators.pop(0)
            docList1 = [docNum for docNum in range(1, 51) if docNum not in docList1]
            result = AndOrOperation(operator, docList1, docList2, result)
            continue

        if operator == "not":                                                                   #handle simple not operation
            docList = [docNum for docNum in range(1, 51) if docNum not in docList1]
            result = docList
        else:
            result = AndOrOperation(operator, docList1, docList2, result)                       #if the operator is and/or call the function

    # print(result)
    return result

def ProximityQuery(query):               #function to handle proximity queries
    words = []
    result = []
    docList = []                         #to store documents where both terms occur
    PositionalDict1 = {}                 #to store positional indexes of first word
    PositionalDict2 = {}                 #to store positional indexes of second word
    k = 0                                #to store the digit
    stemmer = PorterStemmer()            #create object of class PorterStemmer from nltk library

    InvertedIndex = LoadInvertedIndex()             #load inverted index
    PositionalIndex = LoadPositionalIndex()         #load positional index
    queryList = query.split(" ")                    #split query into words

    for word in queryList:                          #processing query to extract words and the digit
        word = word.lower()
        word = re.sub(r"[/]" , "", word)
        if word not in string.whitespace and word.isalpha():
            word = stemmer.stem(word)
            words.append(word)
        elif word.isdigit():
            k = int(word)

    try:
        InvertedIndex1 = InvertedIndex[words[0]][1]
        InvertedIndex2 = InvertedIndex[words[1]][1]
    except KeyError:
        return 0
    else:
        docList = AndOrOperation("and", InvertedIndex1, InvertedIndex2, docList)    #extract the documents where both the terms occur
        PositionalIndex1 = PositionalIndex[words[0]][1]
        PositionalIndex2 = PositionalIndex[words[1]][1]

        for key in PositionalIndex1:                                                #store only those positions of the first word in which the second word also occurs
            if int(key) in docList:
                PositionalDict1[key] = PositionalIndex1[key] 

        for key in PositionalIndex2:                                                #store only those positions of the second word in which the second word also occurs
            if int(key) in docList:
                PositionalDict2[key] = PositionalIndex2[key] 

        for key in PositionalDict1:                                                 #process proximity query
            list1 = PositionalDict1[key]                                            #extract positions document wise
            list2 = PositionalDict2[key]

            for i in list1:                                                         #position of first word in a document
                x = int(i)
                for j in list2:                                                     #position of second word in a document
                    y = int(j)
                    val = abs(x - y)                                                #calculate difference od indexes
                    if val == (k + 1) and key not in result:                        #since second word lies on (k + 1)th index so in between two words we have k words
                        result.append(int(key))

        # print(result)

        return result

def LoadInvertedIndex():
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\InvertedIndex.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        InvertedIndex = json.load(filePtr)             #load inverted index
        filePtr.close()
    
    return InvertedIndex


def LoadPositionalIndex():
    path = pathlib.Path(__file__).parent.absolute()     #get the path of the directory where the script is running
    path = str(path)
    try:
        filePtr = open(path + "\\PositionalIndex.txt", "r", encoding = "utf-8")
    except IOError:
        print("Error: File not found on the specified path.")
    else:
        PositionalIndex = json.load(filePtr)            #load positional index
        filePtr.close()

        return PositionalIndex

# def getQuery():
#     query = 'god and man and love'
#     return query

print("Loading....")  

# PreprocessedData = Preprocessing()
# InvertedIndex = CreateInvertedIndex(PreprocessedData)
# PositionalIndex = CreatePositionalIndex()
# check = re.compile('[/]')
# query = getQuery()
# if check.search(query):
#     result = ProximityQuery(InvertedIndex, PositionalIndex, query)
# else:
#     result = ProcessQuery(InvertedIndex, PositionalIndex, query)
#     print(result)   
# print("Inverted Index: ", InvertedIndex['beyond'])
# print("Positional Index: ", PositionalIndex['beyond'])
