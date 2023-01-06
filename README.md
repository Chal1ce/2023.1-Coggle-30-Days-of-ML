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

# 努力码中。。。。。

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
