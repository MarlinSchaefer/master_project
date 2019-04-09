import keras
import numpy as np
import json
import os
import load_data
import generator as g

def get_model():
    NUM_DETECTORS = 2
    inp_2s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_2s')
    inp_8s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_8s')
    inp_32s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_32s')
    
    batch_1_2s = keras.layers.BatchNormalization()(inp_2s)
    batch_1_8s = keras.layers.BatchNormalization()(inp_8s)
    batch_1_32s = keras.layers.BatchNormalization()(inp_32s)
    
    DROPOUT_RATE = 0.25
    
    dropout_1_2s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_2s)
    dropout_1_8s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_8s)
    dropout_1_32s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_32s)
    
    KERNEL_SIZE = 16
    
    conv_1_2s = keras.layers.Conv1D(KERNEL_SIZE, 32)(dropout_1_2s)
    conv_1_8s = keras.layers.Conv1D(KERNEL_SIZE, 128)(dropout_1_8s)
    conv_1_32s = keras.layers.Conv1D(KERNEL_SIZE, 512)(dropout_1_32s)
    
    activation_1_2s = keras.layers.Activation('relu')(conv_1_2s)
    activation_1_8s = keras.layers.Activation('relu')(conv_1_8s)
    activation_1_32s = keras.layers.Activation('relu')(conv_1_32s)
    
    batch_2_2s = keras.layers.BatchNormalization()(activation_1_2s)
    batch_2_8s = keras.layers.BatchNormalization()(activation_1_8s)
    batch_2_32s = keras.layers.BatchNormalization()(activation_1_32s)

    KERNEL_SIZE = 32
    
    conv_2_2s = keras.layers.Conv1D(KERNEL_SIZE, 32)(batch_2_2s)
    conv_2_8s = keras.layers.Conv1D(KERNEL_SIZE, 128)(batch_2_8s)
    conv_2_32s = keras.layers.Conv1D(KERNEL_SIZE, 512)(batch_2_32s)
    
    activation_2_2s = keras.layers.Activation('relu')(conv_2_2s)
    activation_2_8s = keras.layers.Activation('relu')(conv_2_8s)
    activation_2_32s = keras.layers.Activation('relu')(conv_2_32s)
    
    batch_3_2s = keras.layers.BatchNormalization()(activation_2_2s)
    batch_3_8s = keras.layers.BatchNormalization()(activation_2_8s)
    batch_3_32s = keras.layers.BatchNormalization()(activation_2_32s)
    
    pool_2s = keras.layers.MaxPooling1D(4)(batch_3_2s)
    pool_8s = keras.layers.MaxPooling1D(4)(batch_3_8s)
    pool_32s = keras.layers.MaxPooling1D(4)(batch_3_32s)
    
    flatten_2s = keras.layers.Flatten()(pool_2s)
    flatten_8s = keras.layers.Flatten()(pool_8s)
    flatten_32s = keras.layers.Flatten()(pool_32s)
    
    combined = keras.layers.concatenate([flatten_2s, flatten_8s, flatten_32s])
    
    batch_4 = keras.layers.BatchNormalization()(combined)
    
    dense_1 = keras.layers.Dense(16)(batch_4)
    dense_2 = keras.layers.Dense(1, activation='relu', name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(16)(batch_4)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    model = keras.models.Model(inputs=[inp_2s, inp_8s, inp_32s], outputs=[dense_2, dense_4])
    
    return(model)

def compile_model(model):
    model.compile(loss={'Out_SNR': 'mean_squared_error', 'Out_Bool': 'categorical_crossentropy'}, loss_weights={'Out_SNR': 1.0, 'Out_Bool': 0.5}, optimizer='adam', metrics={'Out_SNR': 'mape', 'Out_Bool': 'accuracy'})

def get_formatted_data(file_path):
    tr_d = format_data_segment(load_data.load_training_data(file_path))
    tr_l = format_label_segment(load_data.load_training_labels(file_path))
    
    te_d = format_data_segment(load_data.load_testing_data(file_path))
    te_l = format_label_segment(load_data.load_testing_labels(file_path))
    
    return(((tr_d, tr_l), (te_d, te_l)))
    

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

def format_data_segment(data):
    tmp = data.transpose((2, 1, 0))
    ret = [np.zeros((2, len(data[0]), len(data))) for i in range(3)]
    ret[0][0] = tmp[1]
    ret[0][1] = tmp[8]
    
    ret[1][0] = tmp[3]
    ret[1][1] = tmp[10]
    
    ret[2][0] = tmp[5]
    ret[2][1] = tmp[12]
    
    ret = [dat.transpose((2, 1, 0)) for dat in ret]
    return(ret)

def format_label_segment(data):
    ret = [[], []]
    for l in data:
        ret[0].append([l[0]])
        ret[1].append([1, 0] if bool(l[1]) else [0, 1])
    ret = [np.array(dat) for dat in ret]
    return(ret)

def train_model(model, data_path, net_path, epochs=None, epoch_break=10, batch_size=32):
    print("Epochs: {}\nEpoch_break={}".format(epochs, epoch_break))
    name = 'collect_net'
    
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
        
        (train_data, train_labels), (test_data, test_labels) = get_formatted_data(data_path)
        
        training_generator = g.DataGeneratorMultInput(train_data, train_labels, batch_size=batch_size)
        testing_generator = g.DataGeneratorMultInput(test_data, test_labels, batch_size=batch_size)
        
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
