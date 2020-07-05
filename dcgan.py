import time, os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from datetime import datetime
from keras.models import Sequential,load_model
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Flatten, BatchNormalization, Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model

# 数据集导入函数
# 参数：数据集路径（推荐绝对路径），batch大小，目标图像尺寸(h,w,c)
# 返回值：一个数据集generator。详见Keras的ImageDataGenerator对象
def load_dataset(dataset_path, batch_size, image_shape):
    dataset_generator = ImageDataGenerator()
    # class_mode置None，不使用标签
    dataset_generator = dataset_generator.flow_from_directory(dataset_path, target_size=(image_shape[0], image_shape[1]),batch_size=batch_size, class_mode=None)
    return dataset_generator

# 搭建鉴别器函数
# 参数：目标图像尺寸(h,w,c)
# 返回值：一个已编译的model
def build_discriminator(image_shape):
    model = Sequential()
    # 第一层卷积
    # 输入大小是image_shape，输出为filters=64通道、大小/strides=2的图像，若除不开则补零（padding='same'）
    # 判别器各卷积层使用he_uniform初始化
    model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform', input_shape=(image_shape)))
    # 判别器各层使用LeakyReLU，防止梯度稀疏
    model.add(LeakyReLU(0.2))
    # 第一层不使用BatchNormalization

    # 第二层卷积
    model.add(Conv2D(filters=128, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))

    # 第三层卷积
    model.add(Conv2D(filters=256, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    # 使用BatchNormalization以提升训练速度
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))

    # 第四层卷积
    model.add(Conv2D(filters=512, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization(momentum=0.5))
    model.add(LeakyReLU(0.2))

    # 第五层全连接输出
    model.add(Flatten())
    model.add(Dense(1))
    # 经测试，使用sigmoid输出性能较好
    # 模型输出值为数值
    model.add(Activation('sigmoid'))

    # 使用Adam优化器
    optimizer = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model

# 搭建生成器函数
# 返回值：一个已编译的model
def build_generator():

    generator = Sequential()

    # 第一层全连接层，输入100通道1*1噪声，输出4*4*512维向量
    generator.add(Dense(units=4 * 4 * 512, kernel_initializer='he_uniform', input_shape=(1, 1, 100)))
    # 使用Reshape把全连接层的输出立体化
    generator.add(Reshape(target_shape=(4, 4, 512)))
    generator.add(BatchNormalization(momentum=0.5))
    # 生成器除输出层外，各层使用relu
    generator.add(Activation('relu'))

    # 第二层反卷积
    generator.add(Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    # 第三层反卷积
    generator.add(Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    # 第四层反卷积
    generator.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    generator.add(BatchNormalization(momentum=0.5))
    generator.add(Activation('relu'))

    # 第五层反卷积
    generator.add(Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', kernel_initializer='glorot_uniform'))
    # 输出层使用tanh
    generator.add(Activation('tanh'))

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)

    return generator


# 保存一个batch生成的图像为一个单一的png
def save_generated_images(generated_images, epoch, batch_idx, save_path):

    plt.figure(figsize=(8, 8), num=2)
    gs1 = gridspec.GridSpec(8, 8)
    gs1.update(wspace=0, hspace=0)

    for i in range(64):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = generated_images[i, :, :, :]
        image += 1
        image *= 127.5
        fig = plt.imshow(image.astype(np.uint8))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    save_name = save_path+'generatedimages_epoch' + str(epoch) + '_batch' + str(batch_idx) + '.png'
    plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    print("Image saved: "+ save_name)

# 建立G+D模型
def build_gan(generator, discriminator):
    model = Sequential()
    # G在前，D在后，G的生成给D去鉴别
    model.add(generator)
    model.add(discriminator)
    optimizer = Adam(lr=0.00015, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model

# 打印训练信息
def display_progress(batch, batch_num, g_loss, d_loss, time_used):
    print("Batch: " + str(batch + 1) + "/" + str(batch_num) +
            " generator loss: " + str(g_loss)+ " discriminator loss: "+ str(d_loss)+
            " Time used: "+ str(time_used) + ' s.')

# 保存loss图像
def save_loss_graph(g_loss_all,d_loss_all,epoch_idx,iter_all,save_path):
    plt.figure(1)
    plt.plot(iter_all, g_loss_all, color='red',label='Generator Loss')
    plt.plot(iter_all, d_loss_all, color='skyblue',label='Discriminator Loss')
    plt.title("DCGAN Train Loss")
    plt.xlabel("Batch Iteration")
    plt.ylabel("Loss")
    if epoch_idx == 0:
        plt.legend()
    plt.savefig('DCGAN_Train_Loss.png')
# 训练函数

def train(dataset_path, image_shape, model_save_path, loss_graph_save_path, generateimage_save_path, epochs = 200, batch_size = 128, is_load = False):
    # 先加载数据生成器
    dataset_generator = load_dataset(dataset_path, batch_size, image_shape)
    # 批次数量
    batch_num = int(DATASET_SIZE / batch_size)
    if is_load == False:

        # 实例化生成器模型
        generator = build_generator()
        # 实例化鉴别器模型
        discriminator = build_discriminator(image_shape)
    else:
        generator = load_model("/home/fulian/DCGAN/DCGAN/models/generator_model_epoch121.hdf5")
        discriminator = load_model("/home/fulian/DCGAN/DCGAN/models/discriminator_model_epoch121.hdf5")
    # 在对抗训练时，只训练生成器，需要把鉴别器冻结
    discriminator.trainable = False
    # 搭建G+D模型用于对抗训练
    gan = Sequential()
    discriminator.trainable = False
    gan.add(generator)
    gan.add(discriminator)

    optimizer = Adam(lr=0.00015, beta_1=0.5)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer,
                metrics=None)
    
    plot_model(generator,to_file=loss_graph_save_path+"generator.png",show_shapes=True)
    plot_model(discriminator,to_file=loss_graph_save_path+"discriminator.png",show_shapes=True)
    plot_model(gan,to_file=loss_graph_save_path+"gan.png",show_shapes=True)

    # 设定几个全局变量
    g_loss_all = np.empty(1)
    d_loss_all = np.empty(1)
    iter_all = np.empty(1)
    iter_idx = 0
    # 开始正式训练，一共epochs轮
    
    for epoch_idx in range(epochs):
        print("Epoch " + str(epoch_idx+1) + "/" + str(epochs) + " :")
        # 每轮训练batch_num个批次
        for batch in range(batch_num):
            start_time = time.time()
            # 读入一批真实样本，数量为batch_size
            real_images = dataset_generator.next()
            # 样本归一化
            real_images = (real_images/127.5)-1
            # 检测一下读入batch的大小。最后一批可能小于给定的batch_size
            current_batch_size = real_images.shape[0]

            # 生成标准正态分布噪声，size=(b,h,w,c)
            noise = np.random.normal(0, 1, size=(current_batch_size,1, 1, 100))
            # 把噪声作为输出送给生成器，获得fake_images
            fake_images = generator.predict(noise)

            # 使用带噪声的标签
            # real_label在0.8到1之间
            real_label = (np.ones(current_batch_size) - np.random.random_sample(current_batch_size) * 0.2)
            # fake_label在0到0.2之间
            fake_label = np.random.random_sample(current_batch_size) * 0.2
            
            # 下面先训练鉴别器
            # 训练鉴别器之前需要先解冻
            discriminator.trainable = True
            # 先用正样本训练
            d_loss_real = discriminator.train_on_batch(real_images, real_label)
            # 再用负样本训练
            d_loss_fake = discriminator.train_on_batch(fake_images, fake_label)
            d_loss = d_loss_real+d_loss_fake
            # 记录loss
            d_loss_all = np.append(d_loss_all, d_loss)
            # 鉴别器训练结束，冻结
            discriminator.trainable = False

            # 下面准备训练生成器
            # 生成标准正态分布噪声，size=(b,h,w,c)；注意这里batch是两倍的（real+fake）
            noise = np.random.normal(0, 1, size=(current_batch_size*2 ,1, 1, 100))

            # 生成一批0.8-1的标签。此时gan中能训练的是生成器，目标就是让gan的输出分数接近输入的标签（高分）
            label = (np.ones(current_batch_size * 2) - np.random.random_sample(current_batch_size * 2) * 0.2)
            g_loss = gan.train_on_batch(noise, label)
            g_loss_all = np.append(g_loss_all, g_loss)

            # 迭代50次就存一次图像
            if((iter_idx + 1) % 50 == 0 and current_batch_size == batch_size):
                save_generated_images(fake_images, epoch_idx, iter_idx, generateimage_save_path)
            time_used = time.time() - start_time

            # 打印训练信息
            display_progress(batch, batch_num, g_loss, d_loss, time_used)
            # 迭代计数器更新
            iter_all = np.append(iter_all, iter_idx)
            iter_idx += 1

        # 每轮保存模型
        # discriminator.trainable = True
        # generator.save(model_save_path+'generator_model_epoch' + str(epoch_idx+1) + '.hdf5')
        # discriminator.save(model_save_path+'discriminator_model_epoch' + str(epoch_idx+1) + '.hdf5')
        # 每轮保存loss图像
        save_loss_graph(g_loss_all,d_loss_all,epoch_idx+1,iter_all,loss_graph_save_path)

def generate(generateimage_save_path, num):
    generator = load_model("/home/fulian/DCGAN/DCGAN/models/generator_model_epoch121.hdf5")
    for i in range(num):
        noise = np.random.normal(0, 1, size=(64,1, 1, 100))
        fake_images = generator.predict(noise)
        time_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_generated_images(fake_images, 0, time_stamp, generateimage_save_path)

if __name__ == "__main__":
    # 数据集样本数
    DATASET_SIZE = 11788
    MODEL_SAVE_PATH = '/home/fulian/DCGAN/DCGAN/models/'
    GENERATE_IMAGE_SAVE_PATH = '/home/fulian/DCGAN/DCGAN/generated_images/'
    DATASET_PATH = '/home/fulian/DCGAN/DCGAN/CUB_200_2011/images/'
    LOSS_GRAPH_SAVE_PATH = '/home/fulian/DCGAN/DCGAN/'
    image_shape = (64, 64, 3)
    train(DATASET_PATH, image_shape, MODEL_SAVE_PATH, LOSS_GRAPH_SAVE_PATH, GENERATE_IMAGE_SAVE_PATH,epochs = 200, batch_size = 64)
    #generate(GENERATE_IMAGE_SAVE_PATH,50)