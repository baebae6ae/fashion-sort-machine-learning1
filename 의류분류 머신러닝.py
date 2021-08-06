#앞에서 만들었던거 사용함
import numpy as np
import matplotlib.pyplot as plt

#의류 이미지 데이터 세트 불러오기
import tensorflow as tf
(x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train_all.shape, y_train_all.shape)
plt.imshow(x_test[1], cmap='gray')
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

x_train = x_train/255
x_val = x_val/255 #이미지는 픽셀마다 0~255 사이의 값을 가지므로 255로 나누면 0~1 사이로 표준화한 것임.

x_train = x_train.reshape(-1, 784)
x_val = x_val.reshape(-1, 784) #이미지가 28x28의 이차원 배열이라 1차원으로 바꾸고 784 길이로 펼치는 것임 (784=28x28)
x_test = x_test.reshape(-1, 784)
print(x_test.shape)


y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_val_encoded = tf.keras.utils.to_categorical(y_val) #타깃 데이터가 0~9 사이의 정수로 확인되기 때문에 원-핫 인코딩으로 10개의 원소를 가진 배열로 바꿔준다 6 = [0.0.0.0.0.0.1.0.0.0.]

class SingleLayer:

  def __init__(self, learning_rate=0.1, l1=0, l2=0):
    self.w = None
    self.b = None
    self.losses = []
    self.val_losses = []
    self.w_history = []
    self.lr = learning_rate
    self.l1 = l1
    self.l2 = l2
  
  def forpass(self, x):
    z=np.dot(x,self.w) + self.b #np.sum말고 np.dot으로 행렬의 곱셈을 함.
    return z

  def backprop(self, x, err):
    m = len(x)
    w_grad = np.dot(x.T, err)/m #x.T는 x행렬을 전치시킨 것(행과 열을 뒤바꿈)으로 err과 곱하고 x 갯수로 나누어 w행렬에 곱할 수 있는 모양으로 만듦.
    b_grad = np.sum(err)/m
    return w_grad, b_grad

  def sigmoid(self, z):
    a = 1/(1+np.exp(-z))
    return a

  def softmax(self,z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1).reshape(-1,1)

#배치경사하강법에서는 전체 샘플을 한꺼번에 계산하므로 for문이 사라짐
  def fit(self, x, y, epochs=100, x_val=None, y_val=None):
    y = y.reshape(-1, 1) #타깃을 열벡터로 바꿈
    y_val = y_val.reshape(-1, 1)
    m = len(x)
    self.w = np.ones((x.shape[1], 1)) #가중치 초기화
    self.b = 0
    self.w_history.append(self.w.copy()) #가중치 기록

    for i in range(epochs):
      z = self.forpass(x)
      a = self.activation(z)
      err = -(y-a)
      w_grad, b_grad = self.backprop(x,err)
      w_grad += (self.l1*np.sign(self.w) + self.l2*self.w) / m
      self.w -= self.lr * w_grad
      self.b -= self.lr*b_grad
      self.w_history.append(self.w.copy())
      a = np.clip(a, 1e-10, 1-1e-10)
      loss = np.sum(-(y*np.log(a) + (1-y)*np.log(1-a)))
      self.losses.append((loss + self.reg_loss()) / m)
      self.update_val_loss(x_val, y_val)

  def predict(self, x):
    z = self.forpass(x)
    c = np.argmax(z)
    print(c)
    return np.argmax(z, axis=1)

  def score(self, x, y):
    return np.mean(self.predict(x) == np.argmax(y, axis=1))

  def reg_loss(self):
    return self.l1*np.sum(np.abs(self.w)) + self.l2/2*np.sum(self.w**2)

  def update_val_loss(self, x_val, y_val):
    z = self.forpass(x_val)
    a = self.softmax(z)
    a = np.clip(a, 1e-10, 1-1e-10)
    val_loss = np.sum(-y_val*np.log(a))
    self.val_losses.append((val_loss + self.reg_loss()) / len(y_val))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) #fit을 이용해 변환규칙을 익힘
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)

class DualLayer(SingleLayer):
  def __init__(self, units=10, batch_size=32, learning_rate=0.1, l1=0, l2=0):
    self.units = units #은닉층의 뉴런 개수
    self.batch_size = batch_size
    self.w1 = None
    self.b1 = None
    self.w2 = None
    self.b2 = None
    self.a1 = None
    self.losses = []
    self.val_losses = []
    self.lr = learning_rate
    self.l1 = l1
    self.l2 = l2

  def forpass(self, x):
    z1 = np.dot(x, self.w1) + self.b1
    self.a1 = self.sigmoid(z1)
    z2 = np.dot(self.a1, self.w2) + self.b2
    return z2
  
  def backprop(self, x, err):
    m = len(x)
    w2_grad = np.dot(self.a1.T, err)/m
    b2_grad = np.sum(err)/m
    #출력층의 가중치와 절편에 대한 그레디언트 계산했음.
    err_to_hidden = np.dot(err, self.w2.T)*self.a1*(1-self.a1)
    #시그모이드함수의 그레디언트 계산
    w1_grad = np.dot(x.T, err_to_hidden)/m
    b1_grad = np.sum(err_to_hidden, axis=0)/m
    #은닉층의 가중치와 절편에 대한 그레디언트 계산했음.
    return w1_grad, b1_grad, w2_grad, b2_grad

  #이제 fit 매서드 만들건데 세개의 메서드로 나누어서 만듦.
  def init_weights(self, n_features): #가중치 초기화 매서드. n_features는 입력 특성의 개수를 지정하는 매개변수
    self.w1 = np.ones((n_features, self.units)) #(특성 개수, 은닉층의 개수)행렬 만들고 전부 1로 채운것
    self.b1 = np.zeros(self.units) #(은닉층의 개수) 행렬 만들고 전부 0으로 채운것
    self.w2 = np.ones((self.units, 1))
    self.b2 = 0

  def fit(self, x, y, epochs=100, x_val=None, y_val=None):
    np.random.seed(42)
    self.init_weights(x.shape[1], y.shape[1])
    for i in range(epochs):
      loss=0
      print('.', end=' ')
      for x_batch, y_batch in self.gen_batch(x, y):
        a=self.training(x_batch, y_batch) #트레이닝 매서드로 따로 나눌거임
        a=np.clip(a, 1e-10, 1-1e-10)
        loss += np.sum(-y_batch*np.log(a))
      self.losses.append((loss + self.reg_loss()) / len(x))
      self.update_val_loss(x_val, y_val)

  def gen_batch(self, x, y):
    length = len(x)
    bins = length // self.batch_size
    if length % self.batch_size:
      bins += 1
    indexes = np.random.permutation(np.arange(len(x)))
    x=x[indexes]
    y=y[indexes]
    for i in range(bins):
      start = self.batch_size * i
      end = self.batch_size * (i+1)
      yield x[start:end], y[start:end]

  def training(self, x, y):
    m = len(x)
    z = self.forpass(x)
    a = self.softmax(z)
    err = -(y-a)
    w1_grad, b1_grad, w2_grad, b2_grad = self.backprop(x,err)
    w1_grad += (self.l1*np.sign(self.w1) + self.l2*self.w1) / m
    w2_grad += (self.l1*np.sign(self.w2) + self.l2*self.w2) / m
    self.w1 -= self.lr * w1_grad
    self.b1 -= self.lr * b1_grad
    self.w2 -= self.lr * w2_grad
    self.b2 -= self.lr * b2_grad
    return a

  def predict(self, x):
    z = self.forpass(x)
    return np.argmax(z, axis=1)

  def reg_loss(self):
    return self.l1*(np.sum(np.abs(self.w1)) + np.sum(np.abs(self.w2))) + self.l2/2*(np.sum(self.w1**2) + np.sum(self.w2**2))

  #나머지 메서드는 싱글레이어와 동일하기 때문에 상속받았으니 생략



#근데 이제 가중치 초기화를 랜덤으로 해야 더 매끄럽게 훈련할 수 있으니 다시 상속받아서 랜덤으로 가중치 초기화하게 만들어보자
class RandomInitNetwork(DualLayer):
  def init_weights(self, n_features, n_classes):
    self.w1 = np.random.normal(0,1,(n_features, self.units))
    self.b1 = np.zeros(self.units)
    self.w2 = np.random.normal(0,1,(self.units,n_classes))
    self.b2 = np.zeros(n_classes)

random_init_net = RandomInitNetwork(units=100, batch_size=256)
random_init_net.fit(x_train, y_train_encoded, x_val=x_val, y_val=y_val_encoded, epochs=40)

random_init_net.score(x_val, y_val_encoded)

class_names = ['티셔츠', '바지', '스웨터', '드레스', '코트', '샌들', '셔츠', '스니커즈', '가방', '앵클부츠']


predictions=random_init_net.predict(x_test)
print('혹시..', end=' ')
print(class_names[predictions[1]], end='')
print('?!')
