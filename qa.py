import numpy as np
from model import QAModel
from data import QAData, Vocabulary
import pickle
import random
from scipy.stats import rankdata
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy import spatial

def main(mode='test', question=None, answers=None,prod_id=1):
    """
    This function is used to train, predict or test

    Args:
        mode (str): train/preddict/test
        question (str): this contains the question
        answers (list): this contains list of answers in string format

    Returns:
        index (integer): index of the most likely answer
    """

    vocabulary = Vocabulary("./data/vocab.txt")
    embedding_file = "./data/Embedding_100dim.txt"
    qa_model = QAModel()
    train_model, predict_model = qa_model.lstm_cnn_attention(embedding_file, len(vocabulary))

    epoch = 1
    if mode == 'train':
        for i in range(epoch):
            print ('Training epoch', i)

            # load training data
            qa_data = QAData()
            questions, good_answers, bad_answers = qa_data.get_training_data()

            # train the model
            Y = np.zeros(shape=(questions.shape[0],))
            train_model.fit(
                [questions, good_answers, bad_answers],
                Y,
                epochs=25,
                batch_size=64,
                validation_split=0.1,
                verbose=1
            )

            # save the trained model
            train_model.save_weights('model/my_train_demo_weights_epoch_' + str(epoch) + '.h5', overwrite=True)
            predict_model.save_weights('model/my_demo_predict_weights_epoch_' + str(epoch) + '.h5', overwrite=True)
    elif mode == 'predict':
        # load the evaluation data
        data = pickle.load(open("./data/dev.pkl",'rb'))
        random.shuffle(data)

        # load weights from trained model
        qa_data = QAData()
        predict_model.load_weights('model/lstm_cnn_attention.h5')

        c = 0
        c1 = 0
        for i, d in enumerate(data):
            print (i, len(data))

            # pad the data and get it in desired format
            indices, answers, question = qa_data.process_data(d)

            # get the similarity score
            sims = predict_model.predict([question, answers])

            n_good = len(d['good'])
            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])
            r = rankdata(sims, method='max')
            c += 1 if max_r == max_n else 0
            c1 += 1 / float(r[max_r] - r[max_n] + 1)

        precision = c / float(len(data))
        mrr = c1 / float(len(data))
        print ("Precision", precision)
        print ("MRR", mrr)
    elif mode == 'test':
        # question and answers come from params
        qa_data = QAData()
        
        question_old=question
        answers, question = qa_data.process_test_data(question, answers)

        # load weights from the trained model
        predict_model.load_weights('model/lstm_cnn_attention.h5')

        # get similarity score
        sims = predict_model.predict([question, answers])
        max_r = np.argmax(sims)
        print(max_r)
        ############################################################################################
        question_answers=pd.read_csv('./data/Questions_Answers.csv')
        question_answers=question_answers[question_answers['Prod_ID']==prod_id]
        question_answers=question_answers.reset_index()
        question_answers=question_answers.drop(columns=['index'])
        
        model1 = SentenceTransformer('bert-base-nli-mean-tokens')
        question_match=pd.DataFrame()
        question_match['score']=np.zeros(len(question_answers['Prod_ID']))
        question_match['Ques_ID']=np.zeros(len(question_answers['Prod_ID']))
        for i in range(0,len(question_answers['Prod_ID'])):
            sentences = []
            sentences.append(question_old)
            sentences.append(str(question_answers['Question_Text'][i]))
            sentence_embeddings = model1.encode(sentences)
            result = 1 - spatial.distance.cosine(sentence_embeddings[0], sentence_embeddings[1])
            #print(result)
            question_match['score'][i]=result
            question_match['Ques_ID'][i]=question_answers['Ques_ID'][i]
            
        question_match=question_match[question_match['score']==max(question_match['score'])]
        
        if question_match['score'].values[0]>=0.75:
            question_out=question_answers[question_answers['Ques_ID']== question_match['Ques_ID'].values[0]]['Question_Text']
            answer_out=question_answers[question_answers['Ques_ID']== question_match['Ques_ID'].values[0]]['Ans_Text']
            found_ques_ans_pair=1
        else:
            question_out='empty'
            answer_out='empty'
            found_ques_ans_pair=0
        
        
        return(max_r,question_out,answer_out,found_ques_ans_pair)
       ############################################################################################

if __name__ == "__main__":
    main(mode='test')

def test(question, answers,prod_id):
    return main(mode='test', question=question, answers=answers,prod_id=prod_id)
