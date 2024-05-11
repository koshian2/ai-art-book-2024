from wordcloud import WordCloud
import matplotlib.pyplot as plt
import glob

def draw_word_cloud(data_directory):
    files = sorted(glob.glob("../data/"+ data_directory + '/*.txt'))
    corpus = []
    for file in files:
        with open(file, 'r') as f:
            words = [x.strip() for x in f.read().strip().split(",")]
            corpus += words

    text = ' '.join(corpus)
    wordcloud = WordCloud(width=1500, height=900, collocations=False,
                          min_font_size=5,
                          background_color ='white').generate(text)

    # ワードクラウドの表示
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")  # 軸を非表示にする
    plt.tight_layout()
    # plt.savefig(f"wordcloud_{data_directory}.png")
    plt.show()

if __name__ == "__main__":
    draw_word_cloud("zunko03")