from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .models import Question, Essay
from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *

import os
current_path = os.path.abspath(os.path.dirname(__file__))


def index(request):
    questions_list = Question.objects.order_by('set')  # 从sql中提取questions数据
    context = {
        'questions_list': questions_list,
    }
    return render(request, 'grader/index.html', context)  # 返回页面index

def essay(request, question_id, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)  # 从sql中提取essay_id的essay数据
    context = {
        "essay": essay,
    }
    return render(request, 'grader/essay.html', context)  # 返回页面essay

def question(request, question_id):
    question = get_object_or_404(Question, pk=question_id)  # 从sql中提取question_id的questions数据
    if request.method == 'POST':
        # 创建一个表单实例，并使用请求中的数据填充它:
        form = AnswerForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data.get('answer')  # 读取name为‘answer’的表单提交值

            if len(content) > 20:  # 文章长度大于20
                num_features = 300
                # 加载训练好的word2vec模型
                model = word2vec.KeyedVectors.load_word2vec_format(os.path.join(current_path, "deep_learning_files/word2vec.bin"), binary=True)

                # 处理content，即输入的文章
                clean_test_essays = []
                clean_test_essays.append(essay_to_wordlist(content, remove_stopwords=True ))
                testDataVecs = getAvgFeatureVecs(clean_test_essays, model, num_features )
                testDataVecs = np.array(testDataVecs)
                testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

                # 构建lstm模型，并加载训练好的数据
                lstm_model = get_model()
                lstm_model.load_weights(os.path.join(current_path, "deep_learning_files/final_lstm.h5"))
                preds = lstm_model.predict(testDataVecs)  # 分数预测值

                if math.isnan(preds):   # 判断预测值是否有效
                    preds = 0
                else:
                    preds = np.around(preds)

                # 限定分数的最大最小值
                if preds < 0:
                    preds = 0
                if preds > 10:
                    preds = 10
            else:  # 若文章长度小于20，分数判为0
                preds = 0

            K.clear_session()
            # 将此文章数据写入sql
            essay = Essay.objects.create(
                content=content,
                question=question,
                score=preds
            )
        # 完整过程：跳转页面到文章页面，在urls内寻找'essay'，再到views内找到对应函数，执行函数
        return redirect('essay', question_id=question.set, essay_id=essay.id)
    else:   # 没有填写文章
        form = AnswerForm()  # 创建空表单实例

    context = {
        "question": question,
        "form": form,
    }
    # 仍返回此页面
    return render(request, 'grader/question.html', context)

