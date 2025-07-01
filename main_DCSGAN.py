import os
import numpy as np
import tensorflow as tf
import pickle
import random
from keras import layers, models, Input
from keras import optimizers
# from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 设置随机种子，保证可重复性
tf.random.set_seed(2)
np.random.seed(2)
random.seed(2)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# 导入数据
train_path = './data/train/C0.pkl'
train_data = load_pkl(train_path)

test_path = './data/test/test_label_1.pkl'
test_data, test_label = load_pkl(test_path)

train_data = train_data[:,:3072]
test_data = test_data[:,:3072]
print(f"训练集形状: {train_data.shape}")
print(f"测试集形状: {test_data.shape}")

# 归一化数据
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# 拓展维度以匹配模型输入
train_data = np.expand_dims(train_data, axis=-1)  # 形状变为 (num_samples, 4096, 1)

# 划分训练集和验证集
x_train, x_val = train_test_split(train_data, test_size=0.2, random_state=2)

# 定义幂等生成网络模型
class Generator(models.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=(3072, 1)),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(16, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(8, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(4, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(2, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
        ])
        self.decoder = models.Sequential([
            layers.Conv1D(2, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1D(4, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1D(8, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1D(16, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1D(32, 3, activation='relu', padding='same'),
            layers.UpSampling1D(2),
            layers.Conv1D(1, 3, activation='sigmoid', padding='same')
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# 修正后的判别器模型
class Discriminator(models.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = models.Sequential([
            layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=(3072, 1)),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(16, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(8, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(4, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Conv1D(2, 3, activation='relu', padding='same'),
            layers.MaxPooling1D(2, padding='same'),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)

# 初始化生成器和判别器
generator = Generator()
discriminator = Discriminator()

# # 打印生成器模型结构
# # generator.build((None, 3072, 1))  # 需要指定输入形状
# generator.model.summary()
#
# # 打印判别器模型结构
# # discriminator.build((None, 3072, 1))  # 需要指定输入形状
# discriminator.model.summary()

# 构建GAN模型
gan_input = Input(shape=(3072, 1))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = models.Model(gan_input, gan_output)

# 编译模型
discriminator.compile(optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9), loss='mse')
gan.compile(optimizer=optimizers.Adam(learning_rate=0.001, beta_1=0.9), loss='mse')

# 训练GAN
epochs = 200
batch_size = 16
half_batch = batch_size // 2

d_losses = []
g_losses = []

for epoch in range(epochs):
    # 训练判别器
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    real_data = x_train[idx]
    noise = np.random.normal(0, 1, (half_batch, 3072, 1))
    generated_data = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_data, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_data, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # 训练生成器
    noise = np.random.normal(0, 1, (batch_size, 3072, 1))
    valid_y = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_y)

    # 保存损失用于可视化
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} [D loss: {d_loss}] [G loss: {g_loss}]")

# 保存损失数据为CSV和Excel
loss_data = {'epoch': list(range(epochs)), 'd_loss': d_losses, 'g_loss': g_losses}
df_loss = pd.DataFrame(loss_data)
df_loss.to_csv('./loss_value/DCSGAN_loss_data.csv', index=False)
df_loss.to_excel('./loss_value/DCSGAN_loss_data.xlsx', index=False)

# 保存损失图
plt.figure(figsize=(10, 8))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('Training and Validation Loss of DCSGAN Model', fontsize=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True)
plt.savefig('./fig/DCSGAN_loss_plot.png')
plt.show()

# 生成样本
noise = np.random.normal(0, 1, (train_data.shape[0], 3072, 1))
generated_data = generator.predict(noise)

# 计算重建误差
train_reconstructed = generator.predict(noise)
print("train_data shape:", train_data.shape)
print("train_reconstructed shape:", train_reconstructed.shape)
train_errors = np.mean(np.square(train_data - train_reconstructed), axis=(1, 2))

# 标准化重建误差
scaler = StandardScaler()
train_errors_scaled = scaler.fit_transform(train_errors.reshape(-1, 1))

# 训练 OC-SVM
oc_svm = svm.OneClassSVM(kernel='rbf', gamma='scale')
oc_svm.fit(train_errors_scaled)

# 测试数据重塑和计算重建误差
test_data = test_data.reshape(-1, 3072, 1)
test_errors = np.mean(np.square(test_data), axis=(1, 2))
test_errors_scaled = scaler.transform(test_errors.reshape(-1, 1))

# 进行异常检测
predictions = oc_svm.predict(test_errors_scaled)
anomalies = np.where(predictions == -1)[0]  # 异常样本索引
print(f"检测到的异常样本数量: {len(anomalies)}")


# 获取决策函数分数
decision_scores = oc_svm.decision_function(test_errors_scaled)



# That's an impressive list of imports.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# 设置全局样式
sn.set_style('whitegrid')
sn.set_palette('muted')
sn.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


# 定义scatter绘图函数
def scatter(x, colors, n_clusters=9):
    palette = np.array(sn.color_palette("hls", n_clusters))
    f, ax = plt.subplots(figsize=(10, 10))

    # 散点图
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int_)])
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    ax.axis('on')
    ax.axis('tight')

    # 标注标签
    txts = {}
    for i in range(n_clusters):
        if i in txts:
            continue
        cluster_center = np.median(x[colors == i, :], axis=0)
        txt = ax.text(cluster_center[0], cluster_center[1], f'C {i}', fontsize=24,
                      bbox=dict(facecolor='none', edgecolor='none'))  # 设置透明底色
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts[i] = txt

    return f, ax, sc, txts


# 使用生成器生成特征进行聚类
train_encoded = generator.encoder.predict(x_train)
test_encoded = generator.encoder.predict(test_data)


# t-SNE降维
def tsne_projection(encoded_data):
    encoded_data = encoded_data.reshape(-1, encoded_data.shape[1])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=2).fit_transform(encoded_data)
    # tsne = TSNE().fit_transform(encoded_data)
    return tsne

# KMeans聚类
def kmeans_clustering(encoded_data, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=2)
    return kmeans.fit_predict(encoded_data)


# 聚类效果评估
# def evaluate_clustering(data, clusters):
#     sil_score = silhouette_score(data, clusters)
#     calinski_score = calinski_harabasz_score(data, clusters)
#     print(f'Silhouette Score: {sil_score:.4f}')
#     print(f'Calinski-Harabasz Index: {calinski_score:.4f}')
def evaluate_clustering(data, clusters):
    sil_score = silhouette_score(data.reshape(data.shape[0], -1), clusters)
    calinski_score = calinski_harabasz_score(data.reshape(data.shape[0], -1), clusters)
    print(f'Silhouette Score: {sil_score:.4f}')
    print(f'Calinski-Harabasz Index: {calinski_score:.4f}')

# 对训练集和测试集的生成特征进行t-SNE降维
train_proj = tsne_projection(train_encoded)
test_proj = tsne_projection(test_encoded)

# 对训练集和测试集进行KMeans聚类
train_clusters = kmeans_clustering(train_proj)
test_clusters = kmeans_clustering(test_proj)

# 可视化聚类结果
plt.figure(figsize=(10, 10))
scatter(test_proj, test_clusters)
plt.title('t-SNE Visualization of DCSGAN Generated Features', fontsize=24)
plt.xlabel('t-SNE Component 1', fontsize=24)
plt.ylabel('t-SNE Component 2', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.grid(True)
plt.savefig('./fig/DCSGAN_cluster_after_encoding.png')
plt.show()

# 评估聚类效果
evaluate_clustering(test_proj, test_clusters)