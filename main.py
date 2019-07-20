from utils import *
import matplotlib.pyplot as plt


#%% Load data
isfloat = True
isNorm = True
data = load_data(isfloat=isfloat, isNorm=isNorm)

#%%
#model = dense_autoencoder(data)
#
## visualize test sample
#test = data['test_flat']
#predict = model.predict(test)
#test = test.reshape(10000,28,28)
#predict = predict.reshape(10000,28,28)
#for i, im in enumerate(test[0:10]):
#    plt.subplot(2, 10, i+1)
#    plt.imshow(test[i])
#    plt.xticks([])
#    plt.yticks([])
#    plt.subplot(2, 10, i+1+10)
#    plt.imshow(predict[i])
#    plt.xticks([])
#    plt.yticks([]) 
#plt.suptitle('Dense and Deep Autoencoder Results: 5 Layers')
#plt.show()
#plt.savefig('dense.png')

#%%
model = cnn_autoencoder(data)
model.summary()

## visualize test sample
test = data['test_img']
test = test.reshape(10000,28,28,1)
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
plt.suptitle('CNN Autoencoder Results')
plt.show()
plt.savefig('cnn.png')
