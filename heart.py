import pandas as pd
import numpy as np
import tensorflow.contrib.keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import initializers
from keras import optimizers
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, ParameterGrid
from sklearn.model_selection import train_test_split


df = pd.read_csv("data.csv", header=None, index_col=False)
df = df.replace('?', np.nan)
for i in range(14):
    median = pd.to_numeric(df[i].dropna()).median()
    df[i] = df[i].fillna(median)


train, test = train_test_split(df, test_size=0.3)

y_train = np.asfarray(train[13].astype('category').to_frame())
x_train = np.asfarray(train.drop([13], axis=1).astype('float32'))
y_test = np.asfarray(test[13].astype('category').to_frame())
x_test = np.asfarray(test.drop([13], axis=1).astype('float32'))

#binary classes
y_train = np.clip(y_train, None, 1)
y_test = np.clip(y_test, None, 1)

# convert class vectors to binary class matrices
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train.shape




epochs = 10
num_input_nodes = 13
batch_size = 5

param_grid = {
    'nodes': [32, 64, 128, 256, 512], 
    'lr': [0.001, 0.01, 0.1, 0.2],  
    'activation1':['relu', 'selu', 'sigmoid'], 
    'activation2':['relu', 'sigmoid', 'softmax'], 
    'loss':['poisson', 'categorical_crossentropy', 'categorical_hinge'],
    'opt_indx':[0],#,1,2], #AdaDelta, Adagrad, RMSprop
    'bias1':[True, False],
    'bias2':[True, False],
}




def build_model(nodes,lr,batch_size,activation1,activation2,loss,opt_indx,bias1,bias2):
    model = Sequential() # means we have layers that are stacked on each other in sequence
    model.add(Dense(nodes, activation=activation1, input_shape=(num_input_nodes,), 
                    use_bias=bias1))
    model.add(Dense(num_classes, activation=activation2, use_bias=bias2))
    
    opt_list = [optimizers.Adadelta(lr=lr), optimizers.Adagrad(lr=lr),optimizers.RMSprop(lr=lr)] 
    model.compile(loss=loss,
                  optimizer=opt_list[opt_indx],
                  metrics=['accuracy'])
    return(model)

model = KerasClassifier(build_fn=build_model, epochs=epochs,batch_size=batch_size,verbose=0)



dist = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=360, cv=3,
                    n_jobs=1, refit=True, verbose=2)
dist_result = dist.fit(x_train, y_train)


# Utility function to report best scores
def report(results, n_top=3):
    with open("hello.txt", "w") as f: 
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                f.write("Model with rank: {0} \n".format(i))
                print("Model with rank: {0}".format(i))
                
                f.write("Mean validation score: {0:.3f} (std: {1:.3f}) \n".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                
                f.write("Parameters: {0} \n".format(results['params'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                
                f.write("\n")
                print("")
report(dist.cv_results_)
