show_n_images = 25
data_dir = './data'

#查看手写数据集的图片
# %matplotlib inline
import os
import tensorflow as tf
from glob import glob
from matplotlib import pyplot
import helper
'''
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
#灰度信息单通道
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')
pyplot.show()  #替代 %matplotlib inline
'''
#查看图片

#
# image_info=helper.get_image(data_dir+ '/mnist/image_0.jpg', 28, 28, 'L')
# print(image_info.shape) #(28,28)
#

# import helper
# data_files=[lists for lists in os.listdir(data_dir+'/img_align_celeba') if os.path.isfile(os.path.join(data_dir+'/img_align_celeba',lists))]
# dataset_name='celeba'
# data=helper.Dataset(dataset_name, data_files)
# print(data.shape) #(60000, 28, 28, 1) 手写数据集

# (41883, 28, 28, 3) 人脸图片
#model_input
#discriminator
#generator
#model_loss
#model_opt
#train

def model_inputs(image_width, image_height, image_channels, z_dim):
	inputs_real=tf.placeholder(tf.float32,(None,image_width,image_height,image_channels),name='input_real')
	inputs_z=tf.placeholder(tf.float32,(None,z_dim),name='input_z')
	# learning_rate=tf.placeholder(tf.float32)
	return inputs_real, inputs_z

#alpha是Leaky ReLU的系数
#用噪声反卷积图片
def generator(z, out_channel_dim, is_train=True,reuse=False,alpha=0.01):
	'''
	:param z:
	:param out_channel_dim:
	:param is_train:
	:param reuse:
	:param alpha:
	:return:
	'''
	with tf.variable_scope('generator',reuse=reuse):
		# 第一层 全连接
		x1=tf.layers.dense(z,7*7*128)
		# 重塑
		x2=tf.reshape(x1,shape=[-1,7,7,128])
		#is_training ,一个占位符储存布尔量，表示网络是否在训练
		x2=tf.layers.batch_normalization(x2,training=is_train)
		x2=tf.nn.leaky_relu(x2,alpha=alpha)
		# Leaky ReLU

		#第二层卷积层
		x3=tf.layers.conv2d_transpose(x2,64,kernel_size=5,strides=2,padding='same')
		x3=tf.layers.batch_normalization(x3,training=is_train)
		x3=tf.nn.leaky_relu(x3,alpha=alpha)
		#[-1,14,14,64]

		#输出层
		logits=tf.layers.conv2d_transpose(
			x3,out_channel_dim,kernel_size=5,strides=2,padding='same'
		)
		#为啥用tanh
		out=tf.tanh(logits) #[-1,28,28,1] out_channel_dim=1
	return out

def discriminator(images, reuse=False,alpha=0.2,is_train=True):
	'''
	:param images:
	:param reuse:
	:param alpha:
	:return:
	'''
	with tf.variable_scope('discriminator',reuse=reuse):
		#第一层不用批归一化
		x1=tf.layers.conv2d(images,64,5,strides=2,padding='same')
		x1=tf.nn.leaky_relu(x1,alpha=alpha)
		#[-1,14,14,128]

		#第二层
		x2=tf.layers.conv2d(x1,128,5,strides=2,padding='same')
		x2=tf.layers.batch_normalization(x2,training=is_train)
		x2=tf.nn.leaky_relu(x2,alpha=alpha)
		#[-1,7,7,64]

		#拉平层
		flatten=tf.reshape(x2,shape=[-1,7*7*128])
		logits=tf.layers.dense(flatten,1)
		out=tf.sigmoid(logits)
	return out, logits

def model_loss(input_real, input_z, out_channel_dim,alpha=0.2, smooth=0.1):
	g_model=generator(input_z,out_channel_dim,alpha=alpha,is_train=True)
	d_model_real,d_logit_real=discriminator(input_real,reuse=False,alpha=alpha)
	d_model_fake,d_logit_fake=discriminator(g_model,reuse=True,alpha=alpha)

	#构建模型损失
	d_loss_real=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_logit_real,labels=tf.ones_like(d_model_real)*(1-smooth)
	))
	d_loss_fake=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_logit_fake,labels=tf.zeros_like(d_model_fake)
	))
	d_loss=d_loss_real+d_loss_fake

	g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		logits=d_logit_fake,labels=tf.ones_like(d_model_fake)
	))

	return d_loss, g_loss

def model_opt(d_loss,g_loss,learning_rate,beta1):
	#获得生成网络和判别网络的变量
	t_vars=tf.trainable_variables()
	g_vars=[var for var in t_vars if var.name.startswith('generator')]
	d_vars=[var for var in t_vars if var.name.startswith('discriminator')]

	#批归一化，需要做一个控制依赖项，确保先更新全局的移动平均的均值和方差，再执行梯度下降
	with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
		d_train_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(d_loss,var_list=d_vars)
		g_train_opt=tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(g_loss,var_list=g_vars)
	return d_train_opt,g_train_opt,g_vars


import numpy as np
def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
	"""
	Show example output for the generator
	:param sess: TensorFlow session
	:param n_images: Number of Images to display
	:param input_z: Input Z Tensor
	:param out_channel_dim: The number of channels in the output image
	:param image_mode: The mode to use for images ("RGB" or "L")
	"""
	cmap = None if image_mode == 'RGB' else 'gray'
	z_dim = input_z.get_shape().as_list()[-1]
	example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

	samples = sess.run(
		generator(input_z, out_channel_dim, is_train=False,reuse=True),
		feed_dict={input_z: example_z})

	images_grid = helper.images_square_grid(samples, image_mode)
	pyplot.imshow(images_grid, cmap=cmap)
	pyplot.show()

from scipy.stats import truncnorm
def train(image_width, image_height, image_channels,epoch_count, batch_size, z_dim, beta1,learning_rate,data,image_mode):
	"""
	Train the GAN
	:param epoch_count: Number of epochs
	:param batch_size: Batch Size
	:param z_dim: Z dimension
	:param learning_rate: Learning Rate
	:param beta1: The exponential decay rate for the 1st moment in the optimizer
	:param get_batches: Function to get batches
	:param data_shape: Shape of the data
	:param data_image_mode: The image mode to use for images ("RGB" or "L")
	"""
	with tf.Graph().as_default():
		input_real,input_z=model_inputs(image_width, image_height, image_channels, z_dim)
		d_loss,g_loss=model_loss(input_real, input_z, image_channels,alpha=0.2, smooth=0.1)
		d_train_opt,g_train_opt,g_vars=model_opt(d_loss,g_loss,learning_rate,beta1)
		saver=tf.train.Saver(max_to_keep=1,var_list=g_vars)
		step=1
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			#获取手写数据集的图片数据
			##(60000, 28, 28, 1)
			# data = helper.Dataset(dataset_name, data_files)

			for e in range(1,epoch_count+1):
				# for ii in range(data.shape[0]//batch_size):
				for x in data.get_batches(batch_size):
					#range[-0.5,0.5]
					x*=2
					batch_z=np.random.uniform(-1,1,size=[batch_size,z_dim])
					sess.run(d_train_opt,{input_real:x,input_z:batch_z})
					sess.run(g_train_opt,{input_real:x,input_z:batch_z})
					# sess.run(d_train_opt,{input_real:X[ii*batch_size:(ii+1)*batch_size],input_z:batch_z})
					# sess.run(g_train_opt,{input_real:X[ii*batch_size:(ii+1)*batch_size],input_z:batch_z})

					if step%2==0:
						train_loss_d=sess.run(d_loss,{input_real:x,input_z:batch_z})
						train_loss_g=sess.run(g_loss,{input_z: batch_z})
						print("Epoch {}/{}...".format(e + 1, epoch_count),
							  "Discriminator Loss: {:.4f}...".format(train_loss_d),
							  "Generator Loss: {:.4f}".format(train_loss_g))


					if step%20==0:
						show_generator_output(sess, show_n_images, input_z, image_channels, image_mode)
				# saver.save(sess, './checkpoints/generator.ckpt')
					step += 1

#超参数设置
#手写数据集channel 1/人脸图片channel 3
image_width, image_height, image_channels=28,28,3
z_dim=100
learning_rate=0.001
batch_size=100
epoch_count=1
alpha=0.2 #leakyRELU的系数
beta1=0.5 #优化器中动量的更新系数


# mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
# with tf.Graph().as_default():
# 	train(image_width, image_height, image_channels,epoch_count, batch_size, z_dim, beta1,learning_rate,data=mnist_dataset,image_mode=mnist_dataset.image_mode)

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
	train(image_width, image_height, image_channels,epoch_count, batch_size, z_dim, beta1,learning_rate,data=celeba_dataset,image_mode=celeba_dataset.image_mode)










