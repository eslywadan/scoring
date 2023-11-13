#文件目录
DATASET_DIR = './data/'
GLOVE_DIR = '.code/glove.6B/'
SAVE_DIR = '.code/'

import os
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec

from keras.layers import Embedding, LSTM, Dense, Dropout, Lambda, Flatten
from keras.models import Sequential, load_model, model_from_config
import keras.backend as K

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score

import nltk

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

nltk.download('stopwords')# 下载停止词，即不能表现内容意义的词，如：'ourselves', 'between', 'but', 'again', 'there'
nltk.download('punkt')# 下载分词工具
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')#加载英文的划分句子的模型(英文句子特点：.之后有空格)

def essay_to_wordlist(essay_v, remove_stopwords):
    """清洗句子/文章，得到句子/文章的词列表"""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)  # 去除文章中非大小写字母以外的字符
    words = essay_v.lower().split() #小写，分词成词列表
    # 去除停止符
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)


def essay_to_sentences(essay_v, remove_stopwords):
    """将文章分句，并调用essay_to_wordlist（）对句子处理"""
    raw_sentences = tokenizer.tokenize(essay_v.strip())#得到句子列表 #strip()去首尾的空格
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """从文章的单词列表中制作特征向量"""
    featureVec = np.zeros((num_features,), dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index_to_key) #训练集中出现的词列表

    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec, model.wv[word])#将每个词向量叠加
    featureVec = np.divide(featureVec, num_words)#文章的特征向量为文章中词向量的平均
    return featureVec


def getAvgFeatureVecs(essays, model, num_features):
    """将文章集生成word2vec模型的词向量"""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays), num_features), dtype="float32")
    # 每篇文章的特征向量
    for essay in essays: # 对每个文章向量化调用makeFeatureVec()向量化
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

def get_model():
    """构建RNN模型"""
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    # 对网络的学习过程进行配置，损失函数为均方误差，评价参数为平均绝对误差
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mae'])
    model.summary()# 输出模型各层的参数状况
    return model
def load_training_data(td='template'):
    if td == 'template': return load_template_training_data()
    if td == 'tayal1': return load_tayal1_training_data()
def load_template_training_data():
    X = pd.read_csv(os.path.join(DATASET_DIR, 'training_set_rel3.tsv'), sep='\t', encoding='ISO-8859-1')  # 读取文件
    X = X.dropna(axis=1)#删除缺省的属性
    X = X.drop(columns=['rater1_domain1', 'rater2_domain1'])#删除各评委的打分

    [r, c] = X.shape
    max_score = [12, 6, 3, 3, 4, 4, 30, 60]
    for i in range(r):
        for j in range(8):
            if X.iloc[i, 1] == j + 1:
                X.iloc[i, 3] =X.iloc[i, 3] /max_score[j]
                print(f"i:{i} / j: {j} / max_score:{max_score[j]} X.iloc[i, 3]: {X.iloc[i, 3]}")
    
    y = X['domain1_score']  # 文章分数y：两位评委对文章的评分和
    return X, y


def load_tayal1_training_data():
    X = pd.read_csv(os.path.join(DATASET_DIR, 'tayal_corpus1.tsv'), sep='\t', encoding='utf-8') 
 
    y = X['score']  # 文章分数y：两位评委对文章的评分和
    return X, y


class TrainScoring():

    def __init__(self,td="template"):
        self.X, self.y = load_training_data(td) 
    def pipe_line(self):
        self.results = []
        self.y_pred_list = []
        count = 1
        X = self.X
        y = self.y
        for traincv, testcv in self.split_data():
            print("\n--------Fold {}--------\n".format(count))
            """划分训练集和测试集"""
            X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]
            train_essays = X_train['essay']
            test_essays = X_test['essay']
            '''word2vec模型'''
            # 从训练集中获取所有句子及分词
            sentences = self.essays_to_sentences(train_essays)
            num_features = 300  # 特征向量的维度
            self.word2vec_model = None
            self.tunned_word2vec_model(sentences, num_features=num_features)
            assert self.word2vec_model is not None
            '''LSTM模型'''
            # 用word2vec模型向量化训练和测试数据中文章
            trainDataVecs = self.essays_to_vecs(train_essays,num_features=num_features)  # 向量化的文章集
            testDataVecs = self.essays_to_vecs(test_essays,num_features=num_features)

            # 将训练向量和测试向量重塑为3维 (1代表一个时间步长)
            trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
            testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

            # 训练lstm模型
            lstm_model = get_model()
            lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=40)
            # lstm_model.load_weights('./model_weights/final_lstm.h5')

            # 使用测试集预测模型输出
            y_pred = lstm_model.predict(testDataVecs)

            # 存储5个模型中最后一个.
            if count == 5:
                lstm_model.save('./model_weights/final_lstm.h5')

            # 评估测试结果
            y_pred = np.around(y_pred) # 将预测值y_pred舍入到最接近的整数
            result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic') # 获取二次均值平均kappa值
            print("Kappa Score: {}".format(result))
            self.results.append(result)
            count += 1


    def split_data(self,n_splits=5, shuffle=True):
        cv = KFold(n_splits=n_splits,shuffle=shuffle)  # 5折交叉验证
        return cv.split(self.X)
    
    def essays_to_sentences(self, essays):
        sentences = []
        # 从训练集中获取所有句子及分词
        for essay in essays:
            sentences += essay_to_sentences(essay, remove_stopwords=True)
        return sentences

    def essays_to_vecs(self, essays, num_features):
        clean_essays = []
        for essay_v in essays:
            clean_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
        DataVecs = getAvgFeatureVecs(clean_essays, self.word2vec_model, num_features)
        return np.array(DataVecs)
    
    def tunned_word2vec_model(self, sentences, num_features=300 ):
        num_features = num_features  # 特征向量的维度
        min_word_count = 40  # 最小词频，小于min_word_count的词被丢弃
        num_workers = 4  # 训练的并行数
        context = 10 # 当前词与预测词在一个句子中的最大距离
        downsampling = 1e-3 # 高频词汇的随机降采样的配置阈值

        # 训练模型
        print("Training Word2Vec Model...")
        model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count=min_word_count, window=context,
                            sample=downsampling)
        model.init_sims(replace=True)  # 结束训练后锁定模型，使模型的存储更加高效
        model.wv.save_word2vec_format('word2vecmodel.bin', binary=True) # 保存模型
        self.word2vec_model = model

    
