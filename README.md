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
        train = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
                sep='\t', names=['query1', 'query2', 'label'])
        valid = pd.read_csv('https://mirror.coggle.club/dataset/LCQMC.test.data.zip', 
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
![image](https://user-images.githubusercontent.com/103374522/210746375-c1cd403d-f5fb-4673-bae3-cdd60b0aa2e7.png)

# 任务2：文本数据分析
