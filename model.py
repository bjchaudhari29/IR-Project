from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, merge, Lambda , Dense, dot , Activation , concatenate, Add , GlobalAveragePooling1D, AveragePooling1D, TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Convolution1D
from keras.models import Model
import numpy as np


def attention_3d_block(hidden_states , question_enc):
    hidden_size = int(hidden_states.shape[2])
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    score_second_part = Dense(hidden_size, use_bias=False, name='attention_score_vec2')(question_enc)
    add_score = Add()([score_first_part, score_second_part])
    tanh_score= Activation('tanh')(add_score)
    attention_weights = Activation('softmax', name='attention_weight')(tanh_score)
    updated_hiddenstates = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    return updated_hiddenstates



class QAModel():
    def get_cosine_similarity(self):
        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())



    def bilstm(self, embedding_file, vocab_size):
        """
        Return the bilstm training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05
        enc_timesteps = 300
        dec_timesteps = 300
        hidden_dim = 64

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        weights = np.loadtxt(embedding_file)
        
        bi_lstm = Bidirectional(LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=True))
        qa_embedding = Embedding(input_dim=vocab_size,output_dim=weights.shape[1],mask_zero=False,weights=[weights])
        # embed the question and pass it through bilstm
        question_embedding =  qa_embedding(question)
        print(question_embedding.shape)
        question_enc_1 = bi_lstm(question_embedding)
        print(question_enc_1.shape)
        question_enc_1 =  GlobalAveragePooling1D()(question_enc_1)
        print(question_enc_1.shape)
        

        # embed the answer and pass it through bilstm
        answer_embedding =  qa_embedding(answer)
        ## Attention 
        print(answer_embedding.shape)
        answer_enc_1 =  bi_lstm(answer_embedding)
        print(answer_enc_1.shape)
        #after_attention = attention_3d_block(answer_enc_1,question_enc_1)
        after_pooling_answer =  GlobalAveragePooling1D()(answer_enc_1)
        print(after_pooling_answer.shape)
        # get the cosine similarity
        similarity = self.get_cosine_similarity()
        
        question_answer_merged = merge(inputs=[question_enc_1, after_pooling_answer], mode=similarity, output_shape=lambda _: (None, 1))
        lstm_model = Model(name="bi_lstm", inputs=[question, answer], outputs=question_answer_merged)
        good_similarity = lstm_model([question, answer_good])
        bad_similarity = lstm_model([question, answer_bad])

        # compute the loss
        loss = merge(
            [good_similarity, bad_similarity],
            mode=lambda x: K.relu(margin - x[0] + x[1]),
            output_shape=lambda x: x[0]
        )

        # return training and prediction model
        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
        prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return training_model, prediction_model
    
    
    def bilstm_attention(self, embedding_file, vocab_size):
        """
        Return the bilstm training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05
        enc_timesteps = 300
        dec_timesteps = 300
        hidden_dim = 128

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        weights = np.loadtxt(embedding_file)
        
        bi_lstm = Bidirectional(LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=True))
        qa_embedding = Embedding(input_dim=vocab_size,output_dim=weights.shape[1],mask_zero=False,weights=[weights])
        # embed the question and pass it through bilstm
        question_embedding =  qa_embedding(question)
        print(question_embedding.shape)
        question_enc_1 = bi_lstm(question_embedding)
        print(question_enc_1.shape)
        question_enc_1 =  GlobalAveragePooling1D()(question_enc_1)
        print(question_enc_1.shape)
        

        # embed the answer and pass it through bilstm
        answer_embedding =  qa_embedding(answer)
        ## Attention 
        print(answer_embedding.shape)
        answer_enc_1 = bi_lstm(answer_embedding)
        after_attention = attention_3d_block(answer_enc_1,question_enc_1)
        after_attention_answer =  GlobalAveragePooling1D()(after_attention)
        print(answer_enc_1.shape)
        # get the cosine similarity
        similarity = self.get_cosine_similarity()
        
        question_answer_merged = merge(inputs=[question_enc_1, after_attention_answer], mode=similarity, output_shape=lambda _: (None, 1))
        lstm_model = Model(name="bi_lstm", inputs=[question, answer], outputs=question_answer_merged)
        good_similarity = lstm_model([question, answer_good])
        bad_similarity = lstm_model([question, answer_bad])

        # compute the loss
        loss = merge(
            [good_similarity, bad_similarity],
            mode=lambda x: K.relu(margin - x[0] + x[1]),
            output_shape=lambda x: x[0]
        )

        # return training and prediction model
        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
        prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return training_model, prediction_model

    def lstm_cnn(self, embedding_file, vocab_size):
        """
        Return the bilstm + cnn training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05
        hidden_dim = 200
        enc_timesteps = 300
        dec_timesteps = 300
        weights = np.loadtxt(embedding_file)

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        # embed the question and answers
        print(vocab_size)
        print(weights.shape)
        qa_embedding = Embedding(input_dim=vocab_size,output_dim=weights.shape[1],weights=[weights])
        question_embedding =  qa_embedding(question)
        answer_embedding =  qa_embedding(answer)
        
        #bi_lstm = Bidirectional(LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=True))

        # pass the question embedding through bi-lstm
        f_rnn = LSTM(hidden_dim, return_sequences=True)
        b_rnn = LSTM(hidden_dim, return_sequences=True)
        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        #print(qb_rnn)
        question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
        print(answer_pool.shape)

        # pass the embedding from bi-lstm through cnn
        cnns = [Convolution1D(filter_length=filter_length,nb_filter=500,activation='tanh',border_mode='same') for filter_length in [1, 2, 3, 5]]
        question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

        # apply max pooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        # get similarity similarity score
        similarity = self.get_cosine_similarity()
        merged_model = merge([question_pool, answer_pool],mode=similarity, output_shape=lambda _: (None, 1))
        lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='lstm_convolution_model')
        good_similarity = lstm_convolution_model([question, answer_good])
        bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
        loss = merge(
            [good_similarity, bad_similarity],
            mode=lambda x: K.relu(margin - x[0] + x[1]),
            output_shape=lambda x: x[0]
        )

        # return the training and prediction model
        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
        prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return training_model, prediction_model



    def lstm_cnn_attention(self, embedding_file, vocab_size):
        """
        Return the bilstm + cnn + attention training and prediction model

        Args:
            embedding_file (str): embedding file name
            vacab_size (integer): size of the vocabulary

        Returns:
            training_model: model used to train using cosine similarity loss
            prediction_model: model used to predict the similarity
        """

        margin = 0.05
        hidden_dim = 200
        enc_timesteps = 300
        dec_timesteps = 300
        weights = np.loadtxt(embedding_file)

        # initialize the question and answer shapes and datatype
        question = Input(shape=(enc_timesteps,), dtype='int32', name='question_base')
        answer = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_good = Input(shape=(dec_timesteps,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(dec_timesteps,), dtype='int32', name='answer_bad_base')

        # embed the question and answers
        print(vocab_size)
        print(weights.shape)
        qa_embedding = Embedding(input_dim=vocab_size,output_dim=weights.shape[1],weights=[weights])
        question_embedding =  qa_embedding(question)
        answer_embedding =  qa_embedding(answer)
        
        #bi_lstm = Bidirectional(LSTM(activation='tanh', dropout=0.2, units=hidden_dim, return_sequences=True))

        # pass the question embedding through bi-lstm
        f_rnn = LSTM(hidden_dim, return_sequences=True)
        b_rnn = LSTM(hidden_dim, return_sequences=True)
        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        #print(qb_rnn)
        question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)
        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)
        answer_pool = attention_3d_block(answer_pool,question_pool)
        #answer_pool = bi_lstm(answer_embedding)
        print(answer_pool.shape)

        # pass the embedding from bi-lstm through cnn
        cnns = [Convolution1D(filter_length=filter_length,nb_filter=500,activation='tanh',border_mode='same') for filter_length in [1, 2, 3, 5]]
        question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

        # apply max pooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        # get similarity similarity score
        similarity = self.get_cosine_similarity()
        merged_model = merge([question_pool, answer_pool],mode=similarity, output_shape=lambda _: (None, 1))
        lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='lstm_convolution_model')
        good_similarity = lstm_convolution_model([question, answer_good])
        bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
        loss = merge(
            [good_similarity, bad_similarity],
            mode=lambda x: K.relu(margin - x[0] + x[1]),
            output_shape=lambda x: x[0]
        )

        # return the training and prediction model
        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
        prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return training_model, prediction_model
    
    