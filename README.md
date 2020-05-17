# IR-Project


Data Preparataion:

1. JSON to CSV Extraction.ipynb - To extract the data in .csv format from the .json files
2. make_GoldStandard.ipynb - To create the Ground Labels for the Training Dataset
3. make_vocab_and_answers_file.ipynb - To create the Vocabulary and the (answer) review files
4. make_embedding_file - To create 100-Dimensional Word2Vec Embeddings for all the words present in our Vocabulary
5. make_train_and_dev_file.ipynb - To create the Training and Validation Datasets


Explanation to all the data files used in the code:

1) vocab.txt: Conatains vocabulary of all the words present in the questions, answers and reviews
2) Embedding_100dim.txt: Word2Vec embedding of all the words present in vocabulary
3) train.pkl: Training dataset; contains list of dictionaries with keys as "question" ans "answers"
	- "question":contains annotated questions according to index of words present in vocabulary
	- "answers": contains indexes of the most relevant reviews to the question in order
4) answers_amazon.pkl: Contains annotated answers according to the index of words present in vocabulary
5) dev.pkl: Validation dataset; conatins list of dictionaries with keys as "bad", "good" and "question"
	- "bad": conatins indexes of the irrelevant reviews to the question
	- "good": contains indexes of the most relevant reviews to the question in order
	- "question": contains annotated questions according to index of words present in vocabulary
	

Introduction to the code files:

1) server.py: to test the code i.e. get answer to any question
2) data.py: contains functions to create data for the model
3) model.py: conatains framework of the models
4) qa.py: contains function to train, test and validate the code
5) Index.html: contains the interface created for the user to enter a question
	
Instructions to get answer to any question or test the model:

1) Run the server.py file 
2) Open a web browser, say Chrome and write "localhost" in the address bar
3) Write your question and product number corresponding to it
4) You will get the answer in a short while

Instructions to train and validate (predict function) the model:

1) Run data.py and model.py
2) Run model.py with "mode" as "train" in main function to train the model or "mode" as "predict" in main function to validate the model
3) To predict using the different models, please load the corresponding model weights and call the corresponding model function in the qa.py file
