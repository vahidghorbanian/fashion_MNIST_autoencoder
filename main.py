from utils import *
import matplotlib.pyplot as plt

#%%
print('Autoencoder applied to Fashion MNIST data set.')
print('This is a classication problem.\n')

#%% Load data
print('============================================')
print('Load data...')
isfloat = True
isNorm = True
data = load_data(isfloat=isfloat, isNorm=isNorm)

#%%
print('============================================')
print('Dense and Fully Connected Autoencoder...')
epoch = 20
model = dense_autoencoder(data,epoch)

# visualize test sample
test = data['test_flat']
predict = model.predict(test)
test = test.reshape(10000,28,28)
predict = predict.reshape(10000,28,28)
for i, im in enumerate(test[0:10]):
    plt.subplot(2, 10, i+1)
    plt.imshow(test[i])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 10, i+1+10)
    plt.imshow(predict[i])
    plt.xticks([])
    plt.yticks([]) 
plt.suptitle('Dense and Deep Autoencoder Results: 5 Layers')
plt.show()
plt.savefig('dense.png')

#%%
print('\n============================================')
print('Dense and Fully Connected Autoencoder...')
epoch = 15
model = cnn_autoencoder(data,epoch)
model.summary()

## visualize test sample
test = data['test_img']
test = test.reshape(10000,28,28,1)
predict = model.predict(test)
test = test.reshape(10000,28,28)
predict = predict.reshape(10000,28,28)
np.save('test_predict', (test, predict))
for i, im in enumerate(test[0:10]):
    plt.subplot(2, 10, i+1)
    plt.imshow(test[i])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 10, i+1+10)
    plt.imshow(predict[i])
    plt.xticks([])
    plt.yticks([]) 
plt.suptitle('CNN Autoencoder Results')
plt.show()
plt.savefig('cnn.png')
