import keras
import numpy as np
import json
import os
import load_data
import generator as g

def incp_lay(x, filter_num):
    l = keras.layers.Conv1D(3 * filter_num, 4, padding='same', activation='relu')(x)
    lm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    lm_2 = keras.layers.Conv1D(2 * filter_num, 8, padding='same', activation='relu')(lm_1)
    rm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    rm_2 = keras.layers.Conv1D(filter_num, 16, padding='same', activation='relu')(rm_1)
    r_1 = keras.layers.MaxPooling1D(4, strides=1, padding='same')(x)
    r_2 = keras.layers.Conv1D(int(round(filter_num)), 1, activation='relu')(r_1)
    
    outp = keras.layers.concatenate([l, lm_2, rm_2, r_2])
    
    return(outp)
    

def get_model():
    inp = keras.layers.Input(shape=(4096,2))
    bn_0 = keras.layers.BatchNormalization()(inp)
    drop_inp = keras.layers.Dropout(0.25)(bn_0)
    
    conv_1 = keras.layers.Conv1D(64, 32, activation='relu')(drop_inp)
    
    bn_1 = keras.layers.BatchNormalization()(conv_1)
    
    pool_0 = keras.layers.MaxPooling1D(4)(bn_1)
    
    conv_2 = keras.layers.Conv1D(128, 16, activation='relu')(pool_0)
    
    bn_2 = keras.layers.BatchNormalization()(conv_2)
    
    incp_1 = incp_lay(bn_2, 32)
    
    incp_2 = incp_lay(incp_1, 32)
    
    pool_2 = keras.layers.MaxPooling1D(2)(incp_2)
    
    incp_3 = incp_lay(pool_2, 32)
    
    incp_4 = incp_lay(incp_3, 32)
    
    incp_5 = incp_lay(incp_4, 32)
    
    incp_6 = incp_lay(incp_5, 32)
    
    norm = keras.layers.BatchNormalization()(incp_6)
    
    pool_4 = keras.layers.MaxPooling1D(4)(norm)
    
    flat = keras.layers.Flatten()(pool_4)
    
    #drop_1 = keras.layers.Dropout(0.4)(flat)
    dense_1 = keras.layers.Dense(8)(flat)
    
    #drop_2 = keras.layers.Dropout(0.4)(dense_1)
    out_snr = keras.layers.Dense(1, name='out_snr', activation='relu')(dense_1)
    
    #drop_3 = keras.layers.Dropout(0.4)(flat)
    dense_2 = keras.layers.Dense(8)(flat)
    
    #drop_4 = keras.layers.Dropout(0.4)(dense_2)
    out_bool = keras.layers.Dense(2,activation='softmax', name='out_bool')(dense_2)
    
    model = keras.models.Model(inputs=[inp], outputs=[out_snr, out_bool])
    
    return(model)

def get_formatted_data(file_path):
    unformatted_tr_data = load_data.load_training_data(file_path)
    unformatted_tr_labels = load_data.load_training_labels(file_path)
    unformatted_te_data = load_data.load_testing_data(file_path)
    unformatted_te_labels = load_data.load_testing_labels(file_path)
    
    tmp = unformatted_tr_data.transpose((2, 1, 0))
    formatted_tr_data = np.zeros((2, len(unformatted_tr_data[0]), len(unformatted_tr_data)))
    formatted_tr_data[0] = tmp[1]
    formatted_tr_data[1] = tmp[8]
    formatted_tr_data = formatted_tr_data.transpose((2, 1, 0))
    
    tmp = unformatted_te_data.transpose((2, 1, 0))
    formatted_te_data = np.zeros((2, len(unformatted_te_data[0]), len(unformatted_te_data)))
    formatted_te_data[0] = tmp[1]
    formatted_te_data[1] = tmp[8]
    formatted_te_data = formatted_te_data.transpose((2, 1, 0))
    
    formatted_tr_labels = [[],[]]
    for l in unformatted_tr_labels:
        formatted_tr_labels[0].append([l[0]])
        formatted_tr_labels[1].append([1, 0] if bool(l[1]) else [0, 1])
    formatted_tr_labels = [np.array(dat) for dat in formatted_tr_labels]
    
    formatted_te_labels = [[],[]]
    for l in unformatted_te_labels:
        formatted_te_labels[0].append([l[0]])
        formatted_te_labels[1].append([1, 0] if bool(l[1]) else [0, 1])
    formatted_te_labels = [np.array(dat) for dat in formatted_te_labels]
    
    #print(formatted_tr_labels)
    
    return(((formatted_tr_data, formatted_tr_labels), (formatted_te_data, formatted_te_labels)))

def compile_model(model):
    model.compile(loss={'out_snr': 'mean_squared_error', 'out_bool': 'categorical_crossentropy'}, loss_weights={'out_snr': 1.0, 'out_bool': 0.5}, optimizer='adam', metrics={'out_snr': 'mape', 'out_bool': 'accuracy'})

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

def train_model(model, data_path, net_path, epochs=None, epoch_break=10, batch_size=32):
    print("Epochs: {}\nEpoch_break={}".format(epochs, epoch_break))
    name = __file__[:-4]
    
    #Store the results of training (i.e. the loss)
    results = []
    
    #Check if epochs is None, if so try to train until the loss of the trainingsset and the one of the testingset seperate by too much
    if epochs == None:
        raise NotImplementedError('Self stopping at overfitting is not implemented.')
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
        
        (train_data, train_labels), (test_data, test_labels) = get_formatted_data(data_path)
        
        training_generator = g.DataGenerator(train_data, train_labels, batch_size=batch_size)
        testing_generator = g.DataGenerator(test_data, test_labels, batch_size=batch_size)
        
        for i in range(ran):
            print("ran: {}\ni: {}".format(ran, i))
            #If epochs were not an integer multiple of epoch_break, the last training cycle has to be smaller
            if i == int(epochs / epoch_break):
                epoch_break = epochs - (ran - 1) * epoch_break
                #Handle the exception of epochs < epoch_break
                if epoch_break < 0:
                    epoch_break += epoch_break
            
            #Fit data to model
            model.fit_generator(generator=training_generator, epochs=epoch_break)
            
            #Iterate counter
            curr_counter += epoch_break
            print(curr_counter)
            
            #Store model after each training-cycle
            model.save(os.path.join(net_path, name + "_epoch_" + str(curr_counter) + ".hf5"))
            print("Stored net")
            
            #Evaluate the performance of the net after every cycle and store it.
            results.append([curr_counter, model.evaluate_generator(generator=training_generator), model.evaluate_generator(generator=testing_generator)])
            #print("Results: {}".format(results))
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    return(model)
