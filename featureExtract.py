

"""Script to run through GENIA tagged training and test files and extract features in preparation for CRF++ NER model training or testing
	
	Written by: Steven Zimmerman
	Date: Mar 6th 2014
	
	
	Description: Originally created as part of an assignment/project for my text analytics course.
	This script can be run on GENIA shared task training or test data (see link below). 
	The input is a collection of PubMed annotated documents (approximately 2000 training). See link below for download of data and further details.
	The output is a file that can be used to train or test a CRF++ model.
	Features created below are based on various reports from the shared task as well as other research papers.  The code can be easily modified to extract different features.
	After extracting features you must create an NER model with CRF++ or some other machine learning algorithm.  Evaluation of the shared task is based on correctly
	determining NER tags in IOB format for biological types including RNA, DNA, gene, cell-type and protein
	Note: CRF = conditional random field CRF++ = is the particular CRF tool I used  NER = named entity recognizer IOB = Inside Outside Beginning
	
	Basic steps that occur with this script: 
	0) Create functions for features to be extracted
	1) Open input training (Genia4ERtask1.iob2) or test (Genia4EReval1.iob2) files 
	2) Create output training (GENIA-CRF-TRAIN.txt) or test (GENIA-CRF-TEST.txt) files
	3) Process each sentence contained in the GENIA input file to extract features for CRF++ training or testing
	4) Write features to CRF++ output files  (see http://crfpp.googlecode.com/svn/trunk/doc/index.html for information about CRF++)
	
	
	Requirements:
	1) This script must be placed in directory containing containing GENIA input file to run correctly
	To download training data tar file from: http://www.nactem.ac.uk/genia/shared-tasks/bionlp-jnlpba-shared-task-2004 
	you will need the Genia4ERtask1.iob2 file specifically
	2) You must have Python 2.7.x with NLTK installed

		
	Resources (very helpful in creation of script): 
	#http://docs.python.org
	#http://docs.python.org/2/library/re.html
	#http://pages.physics.cornell.edu/~myers/teaching/ComputationalMethods/python/arrays.html
	#http://www.nactem.ac.uk/genia/shared-tasks/bionlp-jnlpba-shared-task-2004
	#http://docs.scipy.org/doc/numpy/
	#http://crfpp.googlecode.com/svn/trunk/doc/index.html
	
	TODO:
	1) update script to allow passing in and out of file name rather than manually updating

	Usage:
	$ python featureExtract.py
"""

#####################################
### import existing modules to use###
#####################################
import re   	#for regular expressions
import nltk		#for POS tagging


##############################
### user-defined functions####
##############################
	
########	POS TAGGING 	###############
#Utilizes Python NLTK tagger
def getPOSTags(token):
	#tagger returns a tuple containing original sentence tokens and POS tags
	POSTagsTuple = nltk.pos_tag(token)			
	POSTagsList = []

	#here the POSTags are taken from the tuple and converted to list
	for item in POSTagsTuple:
		POSTagsList.append(item[1])

	return POSTagsList							

	
#######ORTHOGRAPHIC FEATURES #####
#Below are functions to return boolean values for many of the orthographic 
#These features were extracted by almost all research groups in the shared task project
#very straightforward.  I use regular expressions to test
# 1 = true, 0 = false
def getHyphenBool(token):
	#returns bool 1 for true a hyphen is in token
	if re.search("\-",token):
		return 1
	else:
		return 0
		
def getCommaBool(token):
	#returns bool 1 for true a comma is in token
	if re.search(",",token):
		return 1
	else:
		return 0	

def getCapLetterBool(token):
	#returns bool 1 for true a Capital letter is in token
	if re.search("[A-Z]",token):
		return 1
	else:
		return 0		

def getNumberBool(token):
	#returns bool 1 for true a number is in token
	if re.search("[0-9]",token):
		return 1
	else:
		return 0	

def getBackslashBool(token):
	#returns bool 1 for true a backslash is in token
	if re.search('\\\\',token):
		return 1
	else:
		return 0	

def getColonBool(token):
	#returns bool 1 for true a colon is in token
	if re.search(":",token):
		return 1
	else:
		return 0

def getSemiColonBool(token):
	#returns bool 1 for true a semi-colon is in token
	if re.search(";",token):
		return 1
	else:
		return 0

def getBracketBool(token):
	#returns bool 1 for true a bracket is in token
	#only look for [, I assume that if [ exists, then ] also exists
	if re.search('\[',token) or re.search('\]',token):
		return 1
	else:
		return 0		
		
def getParenBool(token):
	#returns bool 1 for true a Paren is in token
	#only look for [, I assume that if ( exists, then ) also exists
	if re.search('\(',token) or re.search('\)',token):
		return 1
	else:
		return 0			
		
	
########	WORD SHAPE 	###############
#I have created a word shape field as a way to normalize the tokens
#Pretty straight forward. Any cap letter is converted to 'A', lower case letters converted to 'a'
#strings of greater than 3 lower case characters are converted to 'aaa'
#digits converted to 'd' and any other character converted to '_'
#This is feature that is frequently applied by multiple research groups in the shared task
def getWordShape(token):
	wordShape = re.sub('[A-Z]', 'A', token, flags=0)
	wordShape = re.sub('[a-z]', 'a', wordShape, flags=0)	
	wordShape = re.sub('aaaa+', 'aaa', wordShape, flags=0)	
	wordShape = re.sub('[0-9]', 'd', wordShape, flags=0)
	wordShape = re.sub('\W', '_', wordShape, flags=0)

	return wordShape

	
#######  LEXICAL INFORMATION   #####
#Based on review of the GENIA training and test data, as well as review of papers in the shared task...
# I have added the following lexical binary features.  These words are strongly correlated with IOB tags
# for instance RNA and transcript are frequently associated with and RNA tag
# The features below are 1 if they exist, else 0
def getGreekLetterBool(token):
	#returns bool 1 for true one of the frequently common greek letter is in token
	if re.search( 'alpha|beta|gamma|delta|epsilon|zeta|theta|kappa|lambda' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0
		
def getRNABool(token):
	#returns bool 1 for true if 'RNA' is string in token
	if re.search( 'rna' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0

def getCellBool(token):
	#returns bool 1 for true  if 'cells' is string in token
	if re.search( 'cell' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0	

def getGeneBool(token):
	#returns bool 1 for true if 'gene' is string in token
	if re.search( 'gene' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0

def getJurkatBool(token):
	#returns bool 1 for true if 'jurkat' is string in token
	if re.search( 'jurkat' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0

def getTranscriptBool(token):
	#returns bool 1 for true if 'transcript' is string in token
	if re.search( 'transcript' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0

def getFactorBool(token):
	#returns bool 1 for true if 'factor' is string in token
	if re.search( 'factor' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0		

def getCommonStringBool(token):
	#returns bool 1 for true if token contains 'prot, mono, nucle, integr, macro or il-'
	if re.search( 'prot|mono|nucle|integr|macro|il\-' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0

def getAnyLexBool(token):
	#returns bool 1 for true if token contains any of the above mentioned lexical features
	if re.search( 'alpha|beta|gamma|delta|epsilon|zeta|theta|kappa|lambda|rna|cell|gene|jurkat|transcript|factor|prot|mono|nucle|integr|macro|il\-' , token , flags = re.IGNORECASE):
		return 1
	else:
		return 0	

#####  IS Capital Letter by itself?   #####
def getCapLetterByselfBool(token):
	#returns bool 1 for true if token is an individual capital letter
	if len(token) == 1 and re.match( '[A-Z]' , token , flags = 0):
		return 1
	else:
		return 0
	

###########################
### Main part of program###
###########################

#FILE Variables: You must manually change these at moment. 
#  For feature extraction of training file Uncomment Genia4ERtask1.iob2 and GENIA-CRF-TRAIN.txt  open file lines, and comment out the other two
#  For feature extraction of test file Uncomment Genia4EReval1.iob2 and GENIA-CRF-TEST.txt  open file lines, and comment out the other two

#TODO: If time permits, change to a function that allows user to pass in file names via command line

#open GENIA input file

#train data file
inFileGenia = open('Genia4ERtask1.iob2','r')

#test data file
#inFileGenia = open('Genia4EReval1.iob2','r')

#open test and training output files
#train output data
oFileTrain = open ('GENIA-CRF-TRAIN.txt','w')   #creates a training csv file for CRF++
 
#test output data
#oFileTrain = open ('GENIA-CRF-TEST.txt','w')   #creates a training csv file for CRF++  


print "Beginning to process GENIA file, number of sentences processed is printed to screen."

#iterate through GENIA input file
#process file one sentence at a time
#get features for each sentence
#write features to output CRF++ file
#print number of sentences processed to screen
#get next sentence
sentenceList = []						#initialize sentence list
IOBList = []							#initialize entity list
tempList = []							#initialize templist to append features for current token, this list gets dumped after each token is written to file
numSentences = 0						#counter for num of sentences processed

for line in inFileGenia:
	#split the current token and entity and load into initial list
	inputTokenEntity = line.split()		
	
	#statement determines whether end of sentence or not.  If not end of sentence, then keep getting tokens to build sentence
	#once sentence is built, then do processing to create feature set for each token
	if len(inputTokenEntity) == 0:
		
		#get the POS Tags for all sentence tokens
		POSTagsList = getPOSTags(sentenceList)
		
		# at this point there is a 
		# sentenceList containing all tokens for the current sentence
		# POSTagsList containing all POS tags associated with each token from current sentence
		# IOBList containing all IOB tags associated with each token from current sentence
		
		#from here the script runs through each token in current sentence to build a list of all features, then writes list/features to file
		#from POSTagsList and sentenceList every other feature can be created
		#loop through all items in sentenceList(tokens) for each POSTag and sentence, 
		# then pass each token through functions to create features and append to tempList
		
		#initialize counter (necessary to append correct POS and IOB tag)
		i = 0
		for token in sentenceList:
			
			#append token and POSTags
			tempList.append(token)
			tempList.append(POSTagsList[i])
			
			#append orographic features
			tempList.append(getHyphenBool(token))
			tempList.append(getCommaBool(token))
			tempList.append(getCapLetterBool(token))
			tempList.append(getNumberBool(token))
			tempList.append(getBackslashBool(token))
			tempList.append(getColonBool(token))
			tempList.append(getSemiColonBool(token))
			tempList.append(getBracketBool(token))
			tempList.append(getParenBool(token))
			
			#append word shape features
			tempList.append(getWordShape(token))
			
			#append Lexical features
			tempList.append(getGreekLetterBool(token))
			tempList.append(getRNABool(token))
			tempList.append(getCellBool(token))
			tempList.append(getGeneBool(token))
			tempList.append(getJurkatBool(token))
			tempList.append(getTranscriptBool(token))
			tempList.append(getFactorBool(token))		
			tempList.append(getCommonStringBool(token))
			tempList.append(getAnyLexBool(token))
			
			#other features
			tempList.append(getCapLetterByselfBool(token))
			
			#append IOBs
			tempList.append(IOBList[i])
			
			#write out token and features to file
			for item in tempList[:-1]:
				oFileTrain.write("%s\t" % item)
			oFileTrain.write(tempList[-1])
			oFileTrain.write("\n")
			
			#clear out tempList and increment current token
			tempList = []
			i = i + 1
		
		#when all tokens and features from current sentence are written to file
		#write another line to denote the space between 2 sentences, 
		#this is helpful for CRF++ and final GENIA shared task evaluation script
		#Then flush out the current sentence and IOB arrays
		oFileTrain.write("\n")
		sentenceList = []				
		IOBList = []

		#increment number of sentences processed and print to screen
		numSentences = numSentences + 1
		print numSentences
		
	else:
		#not end of sentence, so continue to build arrays
		sentenceList.append(inputTokenEntity[0])
		IOBList.append(inputTokenEntity[1])
		
print "Feature Extraction Complete"

###close output files....end of program
inFileGenia.close()
oFileTrain.close()








