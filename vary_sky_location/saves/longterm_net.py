import keras
import numpy as np
import json
import os

"""
def get_model():
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv1D(64, 16, input_shape=(4096,1)))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Conv1D(64, 16))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Conv1D(128, 16))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Conv1D(128, 16))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    
    return(model)
"""    
    

def get_model():
    model = keras.models.Sequential()
    
    model.add(keras.layers.Conv1D(64, 16, input_shape=(4096, 2)))
    model.add(keras.layers.BatchNormalization())
    #Output shape: (None, 4081, 64)
    
    model.add(keras.layers.Permute((2, 1)))
    #Output shape: (None, 64, 4081)
    model.add(keras.layers.Conv1D(64, 4))
    #Output shape: (None, 61, 64)
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(1))
    
    
    #SCALE_FACTOR = 4
    #model = keras.models.Sequential()
 
    #model.add(keras.layers.Conv1D(64/SCALE_FACTOR, 16, input_shape=(4096,2)))
    #model.add(keras.layers.BatchNormalization())
    
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(64))
    #model.add(keras.layers.Dense(4081 * 16))
    #model.add(keras.layers.Reshape((4081, 16)))
    
    ##model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.MaxPooling1D(4))
    ##model.add(keras.layers.Dropout(0.1))
    
    
    ##model.add(keras.layers.Conv1D(128/SCALE_FACTOR, 16))
    ##model.add(keras.layers.BatchNormalization())
    
    #"""
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(128))
    #model.add(keras.layers.Dense(128 / SCALE_FACTOR * 16))
    #model.add(keras.layers.Reshape((128 / SCALE_FACTOR, 16)))
    #"""
    
    ###model.add(keras.layers.BatchNormalization())
    ##model.add(keras.layers.Activation('relu'))
    ##model.add(keras.layers.MaxPooling1D(4))
    ###model.add(keras.layers.Dropout(0.1))
    
    ##model.add(keras.layers.Conv1D(256/SCALE_FACTOR, 16))
    ##model.add(keras.layers.BatchNormalization())
    
    #"""
    #model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dense(256))
    #model.add(keras.layers.Dense(256 / SCALE_FACTOR * 16))
    #model.add(keras.layers.Reshape((256 / SCALE_FACTOR, 16)))
    #"""
    
    ###model.add(keras.layers.BatchNormalization())
    ##model.add(keras.layers.Activation('relu'))
    ##model.add(keras.layers.MaxPooling1D(4))
    ##model.add(keras.layers.Dropout(0.5))
    
    ##model.add(keras.layers.Conv1D(512/SCALE_FACTOR, 8))
    ##model.add(keras.layers.BatchNormalization())
    ###model.add(keras.layers.BatchNormalization())
    ##model.add(keras.layers.Activation('relu'))
    ##model.add(keras.layers.MaxPooling1D(4))
    ##model.add(keras.layers.Dropout(0.5))
    
    ##model.add(keras.layers.Flatten())
    ###model.add(keras.layers.Dense(128,activation='relu'))
    ###model.add(keras.layers.Dropout(0.25))
    ##model.add(keras.layers.Dense(64,activation='relu'))
    ##model.add(keras.layers.Dropout(0.25))
    ##model.add(keras.layers.Dense(1))
    #print(model.summary())
    return(model)

def evaluate_overfitting(train_loss, test_loss):
    THRESHOLD = 0.7
    percentage_loss_difference = [abs(train_loss[i] - test_loss[i]) / train_loss[i] for i in range(len(train_loss))]
    
    bigger_then_threshold_q = [bool(pt > THRESHOLD) for pt in percentage_loss_difference]
    
    if False in bigger_then_threshold_q:
        return(False)
    else:
        return_true = 0
        
        for i in range(len(bigger_then_threshold_q)):
            if return_true == i:
                if i == 0:
                    return_true += 1
                elif test_loss[i] < test_loss[i-1]:
                    return_true += 1
            else:
                return(False)
    
    return(True)

def train_model(model, train_data, train_labels, test_data, test_labels, net_path, epochs=None, epoch_break=10):
    print("Epochs: {}\nEpoch_break={}".format(epochs, epoch_break))
    name = 'longterm_net'
    
    #Store the results of training (i.e. the loss)
    results = []
    
    #Check if epochs is None, if so try to train until the loss of the trainingsset and the one of the testingset seperate by too much
    if epochs == None:
        keepRunning = True
        curr_counter = 0
        
        #Keep training for the number of epochs specified in epoch_break
        while keepRunning:
            #Fit data to model
            model.fit(train_data, train_labels, epochs=epoch_break)
            
            curr_counter += epoch_break
            
            #Save after every training-cycle
            model.save(os.path.join(net_path, name + "_epoch_" + str(curr_counter) + ".hf5"))
            
            #Evaluate the net and store the values
            results.append([curr_counter, model.evaluate(train_data, train_labels), model.evaluate(test_data, test_labels)])
            
            #Train at least 5 times, after that keep training only if no overfitting happens
            if len(results) >= 5:
                start_index = int(curr_counter / epoch_break) - 1
                train_loss = [dat[1][0] for dat in results[start_index:start_index+5]]
                test_loss = [dat[2][0] for dat in results[start_index:start_index+5]]
                keepRunning = not evaluate_overfitting(train_loss, test_loss)
                
    else:
        #Check if epochs are a smiple multiple of epoch_break, meaning that it should train for an integer number of cycles
        if epochs % epoch_break == 0:
            ran = int(epochs / epoch_break)
        #If not, train one more cycle and train only for the left amount of epochs in the last cycle.
        else:
            ran = int(epochs / epoch_break) + 1
        
        #Count how many epochs have passed
        curr_counter = 0
            
        for i in range(ran):
            print("ran: {}\ni: {}".format(ran, i))
            #If epochs were not an integer multiple of epoch_break, the last training cycle has to be smaller
            if i == int(epochs / epoch_break):
                epoch_break = epochs - (ran - 1) * epoch_break
                #Handle the exception of epochs < epoch_break
                if epoch_break < 0:
                    epoch_break += epoch_break
            
            #Fit data to model
            model.fit(train_data, train_labels, epochs=epoch_break)
            
            #Iterate counter
            curr_counter += epoch_break
            print(curr_counter)
            
            #Store model after each training-cycle
            model.save(os.path.join(net_path, name + "_epoch_" + str(curr_counter) + ".hf5"))
            print("Stored net")
            
            #Evaluate the performance of the net after every cycle and store it.
            results.append([curr_counter, model.evaluate(train_data, train_labels), model.evaluate(test_data, test_labels)])
            #print("Results: {}".format(results))
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    return(model)
