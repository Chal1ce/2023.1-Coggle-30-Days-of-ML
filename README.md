# 2023.1-Coggle-医疗搜索相关性比赛打卡

## 背景介绍
文本语义匹配是自然语言处理中一个重要的基础问题，NLP 领域的很多任务都可以抽象为文本匹配任务。例如，信息检索可以归结为查询项和文档的匹配，问答系统可以归结为问题和候选答案的匹配，对话系统可以归结为对话和回复的匹配。语义匹配在搜索优化、推荐系统、快速检索排序、智能客服上都有广泛的应用。如何提升文本匹配的准确度，是自然语言处理领域的一个重要挑战。

- 信息检索：在信息检索领域的很多应用中，都需要根据原文本来检索与其相似的其他文本，使用场景非常普遍。
- 新闻推荐：通过用户刚刚浏览过的新闻标题，自动检索出其他的相似新闻，个性化地为用户做推荐，从而增强用户粘性，提升产品体验。
- 智能客服：用户输入一个问题后，自动为用户检索出相似的问题和答案，节约人工客服的成本，提高效率。


**让我们来看一个简单的例子，比较各候选句子哪句和原句语义更相近**：

- 原句：“车头如何放置车牌”
- 比较句1：“前牌照怎么装”
- 比较句2：“如何办理北京车牌”
- 比较句3：“后牌照怎么装”


**比较结果**：

- 比较句1与原句，虽然句式和语序等存在较大差异，但是所表述的含义几乎相同
- 比较句2与原句，虽然存在“如何” 、“车牌”等共现词，但是所表述的含义完全不同
- 比较句3与原句，二者讨论的都是如何放置车牌的问题，只不过一个是前牌照，另一个是后牌照。二者间存在一定的语义相关性
- 所以语义相关性，句1大于句3，句3大于句2，这就是语义匹配。


## 数据说明
LCQMC数据集比释义语料库更通用，因为它侧重于意图匹配而不是释义。LCQMC数据集包含 260,068 个带有人工标注的问题对。

- 包含 238,766 个问题对的训练集
- 包含 8,802 个问题对的开发集
- 包含 12,500 个问题对的测试集


## 评估方式
使用准确率Accuracy来评估，即：

- 准确率(Accuracy)=预测正确的条目数/预测总条目数

也可以使用文本相似度与标签的皮尔逊系数进行评估，不匹配的文本相似度应该更低。


## 学习打卡任务
![image](https://user-images.githubusercontent.com/103374522/210743695-20cadc7e-c200-4b85-a4a4-d93b21f0cef7.png)



# 任务1：数据集读取
    def load_lcqmc():
        train = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.train.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])
        valid = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.valid.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])
        test = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])
        return train, valid, test
    
    train, valid, test = load_lcqmc()
定义一个函数来读取数据集，并将结果分别返回给train、valid、test。
接下来定义一个函数，简单查看一下训练集、开发集以及测试集的样本。

    def displayhead(dataset1, dataset2, dataset3, n):
        display(dataset1.head(n))
        display(dataset2.head(n))
        display(dataset3.head(n))

    displayhead(train, valid, test, 5)
返回结果如下，从上往下三个表分别对应训练集、开发集以及测试集：
![image](https://user-images.githubusercontent.com/103374522/210924536-8b52b9c3-ab20-466f-9d68-84090e983fec.png)


每个表格中只有三列，前两列为用来进行对比的句子，最后一列为标签。

# 任务2：文本数据分析
## 2.1 缺失值分析
接下来查看缺失值：

    def check_missing(data):
        total = data.isnull().sum().sort_values(ascending = False)
        percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
        missing_data=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data.head()
    
    check_missing(train)
    check_missing(test)
    check_missing(valid)
        
三个表格的返回结果如下图所示

![image](https://user-images.githubusercontent.com/103374522/210751161-ef64c2a0-9433-46dd-b69a-6dd8c4bf5804.png)

可以看到数据集中并不包含缺失值信息。

## 2.2 查看标签分布情况
定义一个函数来讲数据集不同标签的数量计算显示出来

    def label_value_counts(data):
        display(data['label'].value_counts())
        f,ax=plt.subplots(1,2,figsize=(18,8))
        data['label'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.2f%%',ax=ax[0],shadow=True)
        ax[0].set_title('label')
        ax[0].set_ylabel('')
        sns.countplot('label',data=data,ax=ax[1])
        ax[1].set_title('label')
        plt.show()
        
标签数量及标签分布情况：

    label_value_counts(train)
    label_value_counts(valid)
    label_value_counts(test)

执行代码之后三个数据集的标签情况都如下图所示。

训练集标签分布情况如下：

![image](https://user-images.githubusercontent.com/103374522/210925095-cc74b79d-06ca-41b8-8b41-3c5909fdcda0.png)

其中，相似文本对有138574对，不相似文本对有100192对，

开发集标签分布情况如下：

![image](https://user-images.githubusercontent.com/103374522/210925430-e1dfb251-ff39-41bc-a043-b126854cd66b.png)

开发集中，相似文本对有4402对，不相似文本对有4400对，不同标签的文本对数量差异不大

测试集标签分布情况如下：

![image](https://user-images.githubusercontent.com/103374522/210925586-daa9546e-91f5-4da6-bc73-7f735cf74d5a.png)

测试集的相似文本对和不相似文本对都是6250对，不同标签的文本对数量相同

## 2.3 查看文本长度分布情况
使用下面的代码，分析三个数据集的文本长度，并将其可视化：

文本一的分布情况：

    train_query1=train['query1'].str.len()
    valid_query1=valid['query1'].str.len()
    test_query1=test['query1'].str.len()
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,6))
    sns.distplot(train_query1,ax=ax1,color='blue')
    sns.distplot(valid_query1,ax=ax2,color='orange')
    sns.distplot(test_query1,ax=ax3,color='green')
    ax1.set_title('query1 in Train data')
    ax2.set_title('query1 in Valid data')
    ax3.set_title('query1 in Test data')
    plt.show()
![image](https://user-images.githubusercontent.com/103374522/210925688-8ac44987-af79-4353-a2f1-6d66e9cd0614.png)

文本二的分布情况：

    train_query2=train['query2'].str.len()
    valid_query2=valid['query2'].str.len()
    test_query2=test['query2'].str.len()
    fig,(ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,6))
    sns.distplot(train_query2,ax=ax1,color='blue')
    sns.distplot(valid_query2,ax=ax2,color='orange')
    sns.distplot(test_query2,ax=ax3,color='green')
    ax1.set_title('query2 in Train data')
    ax2.set_title('query2 in Valid data')
    ax3.set_title('query2 in Test data')
    plt.show()
![image](https://user-images.githubusercontent.com/103374522/210925736-53f4b577-470d-4752-b4b3-60476c4fa725.png)


可以看到，无论是文本一还是文本二，训练集大多集中在中短文本上，并且训练集的长句的数量以及文本长度要比其他两个集合多。
开发集的文本长度集中在10～15之间，测试集的文本长度则集中在了8～11之间，这两个数据集相比起训练集，长文本的数量都比较少。开发集的文本分布相比起测试集较为集中。

## 2.4 相似文本对与不相似文本对的比较

文本一的标签为0与标签为1的文本长度分布如下图所示：

只需稍微修改上方的绘图代码，即可将结果呈现出来：

    train_query1_0=train['query1'][train['label']==0].str.len()
    valid_query1_0=valid['query1'][valid['label']==0].str.len()
    test_query1_0=test['query1'][test['label']==0].str.len()
    train_query1_1=train['query1'][train['label']==1].str.len()
    valid_query1_1=valid['query1'][valid['label']==1].str.len()
    test_query1_1=test['query1'][test['label']==1].str.len()

    fig,ax=plt.subplots(2,3,figsize=(15,12))
    sns.distplot(train_query1_0,ax=ax[0][0],color='blue')
    sns.distplot(valid_query1_0,ax=ax[0][1],color='orange')
    sns.distplot(test_query1_0,ax=ax[0][2],color='green')
    sns.distplot(train_query1_1,ax=ax[1][0],color='blue')
    sns.distplot(valid_query1_1,ax=ax[1][1],color='orange')
    sns.distplot(test_query1_1,ax=ax[1][2],color='green')

    ax[0][0].set_title('query1 with label-0 in Train data')
    ax[0][1].set_title('query1 with label-0 in Valid data')
    ax[0][2].set_title('query1 with label-0 in Test data')
    ax[1][0].set_title('query1 with label-1 in Train data')
    ax[1][1].set_title('query1 with label-1 in Valid data')
    ax[1][2].set_title('query1 with label-1 in Test data')
    plt.show()
    
![image](https://user-images.githubusercontent.com/103374522/210926903-ddc3e863-6e9e-471d-a26e-d3e36cb9c75c.png)

文本2的标签为0与标签为1的文本长度分布如下图所示：

    train_query2_0=train['query2'][train['label']==0].str.len()
    valid_query2_0=valid['query2'][valid['label']==0].str.len()
    test_query2_0=test['query2'][test['label']==0].str.len()
    train_query2_1=train['query2'][train['label']==1].str.len()
    valid_query2_1=valid['query2'][valid['label']==1].str.len()
    test_query2_1=test['query2'][test['label']==1].str.len()

    fig,ax=plt.subplots(2,3,figsize=(15,12))
    sns.distplot(train_query2_0,ax=ax[0][0],color='blue')
    sns.distplot(valid_query2_0,ax=ax[0][1],color='orange')
    sns.distplot(test_query2_0,ax=ax[0][2],color='green')
    sns.distplot(train_query2_1,ax=ax[1][0],color='blue')
    sns.distplot(valid_query2_1,ax=ax[1][1],color='orange')
    sns.distplot(test_query2_1,ax=ax[1][2],color='green')

    ax[0][0].set_title('query2 with label-0 in Train data')
    ax[0][1].set_title('query2 with label-0 in Valid data')
    ax[0][2].set_title('query2 with label-0 in Test data')
    ax[1][0].set_title('query2 with label-1 in Train data')
    ax[1][1].set_title('query2 with label-1 in Valid data')
    ax[1][2].set_title('query2 with label-1 in Test data')
    plt.show()

![image](https://user-images.githubusercontent.com/103374522/210926979-13889266-a5ea-4938-9338-ed1e9db86806.png)

对于文本一，从数据发布上来看，相似文本对（标签为1）在短文本上的数量会比不相似文本对（标签为0）多，而在长度适中的文本上，相似文本对的文本平均长度要比不相似文本对的平均文本长度长。

对于文本二，不相似文本对的文本长度分布明显比相似文本对的长度分布要均衡一些。

## 2.5 词频统计分析

### 2.5.1 训练集结果

    txt = open("train.csv", "r", encoding='utf-8').read()
    words = jieba.lcut(txt)
    counts = {}
    for word in words:
        if  len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word, 0) + 1

    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    for i in range(20):
        word, count = items[i]
        print("{0:<5}:{1:>5}".format(word, count))

输出训练集出现频率最高的20个词，如图所示：

![image](https://user-images.githubusercontent.com/103374522/211007785-0a3d0abc-8691-4d77-9e8b-71591874d15c.png)

统计输出训练集所有文本中共有1808847个词：

    sum_num = 0
    for i in range(len(items)):
        word, count = items[i]
        sum_num += count
    print("训练集的词数量共有{}个。".format(sum_num))

![image](https://user-images.githubusercontent.com/103374522/211008106-a7943fab-24d3-4071-90b5-40b9cee74c5b.png)

再通过词云进行可视化查看：

    wc = WordCloud(font_path='../input/fonts-on-mac/Fonts/方正正中黑简体.TTF',background_color="white")
    wc.generate_from_frequencies(counts)

    wc.to_file("Training data wordcloud.png")

    plt.figure(figsize=(8,4))
    plt.imshow(plt.imread("./Training data wordcloud.png"))
    plt.axis("off")
    plt.show()
    
![image](https://user-images.githubusercontent.com/103374522/211008368-44ee25df-f128-4598-b1a1-8f9d61fdd10a.png)

### 2.5.2 开发集的结果

出现频率最高的20个词：

    txt = open("valid.csv", "r", encoding='utf-8').read()
    words = jieba.lcut(txt)
    counts = {}
    for word in words:
        if  len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word, 0) + 1

    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    for i in range(20):
        word, count = items[i]
        print("{0:<5}:{1:>5}".format(word, count))

![image](https://user-images.githubusercontent.com/103374522/211008499-a1139814-1244-4883-99b3-68f6a13af8b9.png)

统计输出开发集所有文本中共有76293个词：

    sum_num = 0
    for i in range(len(items)):
        word, count = items[i]
        sum_num += count
    print("开发集的词数量共有{}个。".format(sum_num))
    
![image](https://user-images.githubusercontent.com/103374522/211008624-92ae6d9d-35e3-4332-bec7-812eff611cda.png)

开发集的词云：

![image](https://user-images.githubusercontent.com/103374522/211008673-57cdf428-f05b-4887-b364-df69b1d10904.png)

### 2.5.3 测试集结果

输出测试集出现次数最多的20个词：

    txt = open("test.csv", "r", encoding='utf-8').read()
    words = jieba.lcut(txt)
    counts = {}
    for word in words:
        if  len(word) == 1:
            continue
        else:
            counts[word] = counts.get(word, 0) + 1

    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)

    for i in range(20):
        word, count = items[i]
        print("{0:<5}:{1:>5}".format(word, count))
        
![image](https://user-images.githubusercontent.com/103374522/211008868-19d4771a-a3ab-4a97-ba1f-00a7733e8c6c.png)

统计测试集的词数量：

    sum_num = 0
    for i in range(len(items)):
        word, count = items[i]
        sum_num += count
    print("开发集的词数量共有{}个。".format(sum_num))

![image](https://user-images.githubusercontent.com/103374522/211008998-958a3528-f333-48f0-bede-99d57f06a086.png)

测试集的词云如图：

    wc = WordCloud(font_path='../input/fonts-on-mac/Fonts/方正正中黑简体.TTF',background_color="white")
    wc.generate_from_frequencies(counts)

    wc.to_file("Testing data wordcloud.png")

    plt.figure(figsize=(8,4))
    plt.imshow(plt.imread("./Testing data wordcloud.png"))
    plt.axis("off")
    plt.show()

![image](https://user-images.githubusercontent.com/103374522/211009329-7e9113e8-f8aa-41d3-91e1-b15973ebd0d8.png)

## 2.6 字符分布
    train_qs = pd.Series(train['query1'].tolist() + train['query2'].tolist()).astype(str)
    valid_qs = pd.Series(valid['query1'].tolist() + valid['query2'].tolist()).astype(str)
    test_qs = pd.Series(test['query1'].tolist() + test['query2'].tolist()).astype(str)

    dist_train = train_qs.apply(len)
    dist_valid = valid_qs.apply(len)
    dist_test = test_qs.apply(len)

    plt.figure(figsize=(15, 10))
    plt.hist(dist_train, bins=50, range=[0, 50], label='Train')
    plt.hist(dist_valid, bins=50, range=[0, 50], label='Valid')
    plt.hist(dist_test, bins=50, range=[0, 50], label='Test')
    plt.title("Characters count in the dataset")
    plt.legend()
    
输出结果如下，蓝色为训练集，绿色为测试集，橙色为开发集：

![image](https://user-images.githubusercontent.com/103374522/211142079-25d2d1f3-7457-4633-aa94-44c17ab36bd8.png)

# 任务3：文本相似度（统计特征）
## 3.1 计算文本长度
    def question_len(data):
        data['q1_length'] = data['query1'].apply(lambda x:len(x))
        data['q2_length'] = data['query2'].apply(lambda x:len(x))

    question_len(train)
    question_len(valid)
    question_len(test)

通过运行上述代码，将数据集中的问题进行逐行分析，并且将结果保存在数据集中，输出结果如下，从上往下依次是：训练集、开发集、测试集：

    display(train.head())
    display(valid.head())
    display(test.head())
![image](https://user-images.githubusercontent.com/103374522/211142563-41feecd3-9dc1-4580-afa7-24ecddc7e16a.png)

## 3.2 统计文本单词个数
    def question_count(data):
        data['q1_count'] = data['query1'].apply(lambda x:len(jieba.lcut(x)))
        data['q2_count'] = data['query2'].apply(lambda x:len(jieba.lcut(x)))

    question_count(train)
    question_count(valid)
    question_count(test)
    
将文本用jieba库进行切割，并且使用len()函数计算文本的单词个数。

    display(train.head())
    display(valid.head())
    display(test.head())

最终输出结果如下：

![image](https://user-images.githubusercontent.com/103374522/211142796-ffccce15-33dc-4f09-adbb-ae0c710942f9.png)

## 3.3 文本单词差异
    def question_compare(data):
        data['q1_words'] = data['query1'].apply(lambda x:jieba.lcut(x))
        data['q2_words'] = data['query2'].apply(lambda x:jieba.lcut(x))
        data[['common_words', 'q1_new', 'q2_new']] = ' '
        for i in range(len(data['query1'])):
            ls1 = data['q1_words'][i]
            ls2 = data['q2_words'][i]
            common = set(ls1).intersection(set(ls2))
            new_ls1 = ' '.join([w for w in ls1 if w not in common])
            new_ls2 = ' '.join([w for w in ls2 if w not in common])
            data['common_words'][i] = list(common)
            data['q1_new'][i] = list(new_ls1)
            data['q2_new'][i] = list(new_ls2)
        data['common_words_len'] = data['common_words'].apply(lambda x:len(x))
        data['q1_new_len'] = data['q1_new'].apply(lambda x:len(x))
        data['q2_new_len'] = data['q2_new'].apply(lambda x:len(x))
        display(data.head())
         
    question_compare(train)
    question_compare(valid)
    question_compare(test)

对文本1和文本2用jieba库进行分词，统计相同的词以及不同的词，以及这些词的数量，并且将其存储在数据集中，将其输出如下图所示：

![image](https://user-images.githubusercontent.com/103374522/211150965-85aa7d1c-6722-41a1-ad63-e08eea94521c.png)

![image](https://user-images.githubusercontent.com/103374522/211150982-45e1c514-ca09-4cb5-a37c-ce74c5fdd82e.png)

![image](https://user-images.githubusercontent.com/103374522/211151003-be971a51-dd93-4e53-8134-98259044d47f.png)

## 3.5 最长公用字符串长度

    def max_len(ls):
        max_lens = 0
        for i in ls:
            max_lens = len(i) if len(i) > max_lens else max_lens

        return max_lens

    def max_common(data):
        data['common_words_max'] = ''
        for i in range(len(data['query1'])):
            data['common_words_max'][i] = max_len(data['common_words'][i])
        display(data.head())

    max_common(train)
    max_common(valid)
    max_common(test)
    
在这里分别定义两个函数，第一个函数是接收传过来的共同的单词数列，然后计算出最长的公用字符串长度，第二个函数是用来便利每一个公共词数列，将其传入第一个函数中，并且将结果保存在数据集之中，函数运行结果如下，从上到下分别为训练集、开发集、测试集。

![image](https://user-images.githubusercontent.com/103374522/211151903-ce023b33-432b-43de-b810-13a2e4103847.png)

![image](https://user-images.githubusercontent.com/103374522/211151963-174fe5bf-8b96-4185-9100-becc33ce6a94.png)

![image](https://user-images.githubusercontent.com/103374522/211151953-086e096f-69d5-4d4c-aec3-0bb76b7a18ae.png)

## 3.6 TFIDF文本相似度

在这里，使用TF-IDF构建稀疏特征，并使用pickle对生成的特征进行缓存，得到文本1和文本2的TF-IDF稀疏特征。

    from sklearn.feature_extraction.text import TfidfVectorizer
    len_train = train.shape[0]

    data_all = pd.concat([train, valid, test])

    corpus=[]
    max_features = 40
    ngram_range = (1, 2)
    min_df = 3
    print("Generate tfidf")
    feats = ['query1', 'query2']
    vect_orig = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)

    for f in feats:
        data_all[f] = data_all[f].astype(str)
        corpus += data_all[f].values.tolist()

    vect_orig.fit(corpus)

    for f in feats:
        train_tfidf = vect_orig.transform(train[f].astype(str).values.tolist())
        valid_tfidf = vect_orig.transform(valid[f].astype(str).values.tolist())
        test_tfidf = vect_orig.transform(test[f].astype(str).values.tolist())
        pd.to_pickle(train_tfidf, "train_%s_tfidf_v2.pkl"%f)
        pd.to_pickle(valid_tfidf, "valid_%s_tfidf_v2.pkl"%f)
        pd.to_pickle(test_tfidf, "test_%s_tfidf_v2.pkl"%f)
        
使用生成的TF-IDF文件进行相似度特征构建，在这里相似度类型选用余弦夹角。
 
    from sklearn.metrics.pairwise import pairwise_distances

    def calc_cos_dist(query1, query2, metric='cosine'):
        return pairwise_distances(query1, query2, metric=metric)[0][0]

    train_q1_tfidf = pd.read_pickle("train_query1_tfidf_v2.pkl")
    train_q2_tfidf = pd.read_pickle("train_query2_tfidf_v2.pkl")
    valid_q1_tfidf = pd.read_pickle("valid_query1_tfidf_v2.pkl")
    valid_q2_tfidf = pd.read_pickle("valid_query2_tfidf_v2.pkl")
    test_q1_tfidf = pd.read_pickle("test_query1_tfidf_v2.pkl")
    test_q2_tfidf = pd.read_pickle("test_query2_tfidf_v2.pkl")

    train_tfidf_sim = []
    for r1, r2 in zip(train_q1_tfidf, train_q2_tfidf):
        train_tfidf_sim.append(calc_cos_dist(r1,r2))

    valid_tfidf_sim = []
    for r1, r2 in zip(valid_q1_tfidf, valid_q2_tfidf):
        valid_tfidf_sim.append(calc_cos_dist(r1,r2))

    test_tfidf_sim = []
    for r1, r2 in zip(test_q1_tfidf, test_q2_tfidf):
        test_tfidf_sim.append(calc_cos_dist(r1,r2))

    train_tfidf_sim = np.array(train_tfidf_sim)
    valid_tfidf_sim = np.array(valid_tfidf_sim)
    test_tfidf_sim = np.array(test_tfidf_sim)
    
![image](https://user-images.githubusercontent.com/103374522/211155182-4c27f8b9-d924-47ed-a58d-c14bff90a4b1.png)

从相似度标签来看，词组的差异以及共有的词的数量是最具有区分性的两个特征，通过词的差异，我们能看到，当标签为1，即为相似文本对时，文本1和文本2出现的不同的词很少，文本1和文本2在词组上没有很大的差异性，并且拥有着较多的共同词，反之文本1和文本2出现不同的词语多，共同的词语数量少。

# 任务4:文本相似度（词向量与句子编码）
## 4.1 使用word2vec训练词向量
在上述使用jieba库分词的基础上，训练一个w2v模型，并且通过w2v进行词嵌入

由于前面已经使用了jieba库进行分词，已经生成了一个词汇列表，故在此可以不用再定义一个函数text_to_word_list(text)去进行数据处理
    
    def text_to_word_list(text)
    
定义一个函数来将其两个文本划分为字典。

    def split_and_zero_padding(df, max_seq_length):
        X = {'left': df['q1_words_n'], 'right': df['q2_words_n']}

        # 另填充
        for dataset, side in itertools.product([X], ['left', 'right']):
            dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_seq_length)

        return dataset

训练一个w2v模型，并将其保存。

    def extract_questions():

        df1 = pd.read_csv("train.csv")
        df2 = pd.read_csv("valid.csv")
        df3 = pd.read_csv("test.csv")

        for dataset in [df1, df2, df3]:
            for i, row in dataset.iterrows():
                if i != 0 and i % 1000 == 0:
                    logging.info("read {0} sentences".format(i))

                if row['query1']:
                    yield gensim.utils.simple_preprocess(row['query1'])
                if row['query2']:
                    yield gensim.utils.simple_preprocess(row['query2'])


    documents = list(extract_questions())
    logging.info("Done reading data file")

    model = gensim.models.Word2Vec(documents, size=300)
    model.train(documents, total_examples=len(documents), epochs=10)
    model.save("Query1-Quesry2.w2v")

使用上述保存的模型进行词嵌入。

    def make_w2v_embeddings(df, embedding_dim=300, empty_w2v=False):
        vocabs = {}
        vocabs_cnt = 0
        vocabs_not_w2v = {}
        vocabs_not_w2v_cnt = 0
        # 停用词
        stops = set(stopwords.words('chinese'))

        if empty_w2v:
            word2vec = EmptyWord2Vec
        else:
            word2vec = gensim.models.word2vec.Word2Vec.load("Query1-Quesry2.w2v").wv

        for index, row in df.iterrows():
            # 遍历该行的两个文本
            for query in ['q1_words', 'q2_words']:
                q2n = []
                for word in text_to_word_list(row[query]):
                    # 查看是否为停用词
                    if word in stops:
                        continue                   
                    if word not in word2vec.vocab:
                        if word not in vocabs_not_w2v:
                            vocabs_not_w2v_cnt += 1
                            vocabs_not_w2v[word] = 1

                    # 如果这是之前没有见过的词，就把它附在词汇字典里。
                    if word not in vocabs:
                        vocabs_cnt += 1
                        vocabs[word] = vocabs_cnt
                        q2n.append(vocabs_cnt)
                    else:
                        q2n.append(vocabs[word])

                # 将它存放在新的一列之中
                df.at[index, query + '_n'] = q2n

        # 嵌入矩阵
        embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)
        embeddings[0] = 0 

        # 构建嵌入矩阵
        for word, index in vocabs.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        del word2vec
        return df, embeddings

处理之后的训练集如图所示：

![image](https://user-images.githubusercontent.com/103374522/211251650-7a1a1bc5-70c3-4e18-8feb-d2dd1b74d9bc.png)

# 任务5：LSTM孪生网络

Keras自定义层，计算曼哈顿距离。

    class ManDist(Layer):
        def __init__(self, **kwargs):
            self.result = None
            super(ManDist, self).__init__(**kwargs)

        def build(self, input_shape):
            super(ManDist, self).build(input_shape)

        # 这是该层的逻辑所在。
        def call(self, x, **kwargs):
            self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
            return self.result

        # 返回output_shape
        def compute_output_shape(self, input_shape):
            return K.int_shape(self.result)


定义一些模型的参数

    gpus = 1
    batch_size = 256
    n_epoch = 20
    n_hidden = 100
    
开始定义模型

    x = Sequential()
    x.add(Embedding(len(embeddings), embedding_dim,
                    weights=[embeddings], input_shape=(max_seq_length,), trainable=False))
    x.add(LSTM(n_hidden))
    shared_model = x
    
输入层

    left_input = Input(shape=(max_seq_length,), dtype='int32')
    right_input = Input(shape=(max_seq_length,), dtype='int32')
    
将其全部打包成曼哈顿距离模型

    malstm_distance = ManDist()([shared_model(left_input), shared_model(right_input)])
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    
准备开始训练模型

    training_start_time = time()
    malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=([X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                            training_end_time - training_start_time))

![image](https://user-images.githubusercontent.com/103374522/211257950-8a887406-7992-48fe-b7cf-0567cbacd5e2.png)


保存训练完的模型

![image](https://user-images.githubusercontent.com/103374522/211261549-8f1d404a-6031-47ee-9289-d8e3d3fa331f.png)

训练20轮完毕，保存模型

    model.save('SiameseLSTM.h5')
    
训练过程的准确率和loss

    plt.subplot(211)
    plt.plot(malstm_trained.history['accuracy'])
    plt.plot(malstm_trained.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('history-graph.png')
 
![image](https://user-images.githubusercontent.com/103374522/211264060-1e5ec074-cb3d-45f0-a295-dadabc8f5063.png)

开始对测试集进行预测

    test_df = test
    for q in ['q1_words', 'q2_words']:
        test_df[q + '_n'] = test_df[q]

    embedding_dim = 300
    max_seq_length = 40
    test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

    X_test = split_and_zero_padding(test_df, max_seq_length)
    y_true = test_df['label']
    model = load_model('SiameseLSTM.h5', custom_objects={'ManDist': ManDist})

    prediction = model.predict([X_test['left'], X_test['right']])
    prediction = np.argmax(prediction, axis=1)

    print('Accuracy:{}'.format(accuracy_score(y_true, prediction)))
    
<img width="212" alt="image" src="https://user-images.githubusercontent.com/103374522/211315337-dc249daa-3b9d-4356-b9e8-cf1613b5ebd8.png">


# 任务6:SBERT模型

