import DataPretreat
import Word2Vec

def main():
    print("数据预处理阶段")
    DataPretreat.prepare_data(windows_size=3)
    vocabulary_size = len(DataPretreat.vocabulary_list)
    SkipGram = Word2Vec.SkipGram(vocabulary_size)
    print("创建SkipGram神经网络")
    SkipGram.build_network()
    print("训练SkipGram神经网络")
    SkipGram.train()
    print("可视化SkipGram训练效果")
    SkipGram.visualize()



if __name__ == "__main__":
    main()