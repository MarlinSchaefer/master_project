import keras
import numpy as np
import json
import os
import load_data
import h5py
import generator as g

def get_model():
    inp = keras.layers.Input(shape=(4096,14))
    bn_0 = keras.layers.BatchNormalization()(inp)
    
    conv_1 = keras.layers.Conv1D(32, 16)(bn_0)
    bn_1 = keras.layers.BatchNormalization()(conv_1)
    act_1 = keras.layers.Activation('relu')(bn_1)
    pool_1 = keras.layers.MaxPooling1D(4)(act_1)
    
    conv_2 = keras.layers.Conv1D(64, 16)(pool_1)
    bn_2 = keras.layers.BatchNormalization()(conv_2)
    act_2 = keras.layers.Activation('relu')(bn_2)
    pool_2 = keras.layers.MaxPooling1D(4)(act_2)
    
    conv_3 = keras.layers.Conv1D(128, 16)(pool_2)
    bn_3 = keras.layers.BatchNormalization()(conv_3)
    act_3 = keras.layers.Activation('relu')(bn_3)
    pool_3 = keras.layers.MaxPooling1D(4)(act_3)
    
    conv_4 = keras.layers.Conv1D(256, 16)(act_3)
    bn_4 = keras.layers.BatchNormalization()(conv_4)
    act_4 = keras.layers.Activation('relu')(bn_4)
    pool_4 = keras.layers.MaxPooling1D(4)(act_4)
    
    flat = keras.layers.Flatten()(pool_4)
    
    drop_1 = keras.layers.Dropout(0.4)(flat)
    dense_1 = keras.layers.Dense(64)(drop_1)
    
    drop_2 = keras.layers.Dropout(0.4)(dense_1)
    out_snr = keras.layers.Dense(1, name='out_snr')(drop_2)
    
    drop_3 = keras.layers.Dropout(0.4)(flat)
    dense_2 = keras.layers.Dense(16)(drop_3)
    
    drop_4 = keras.layers.Dropout(0.4)(dense_2)
    out_bool = keras.layers.Dense(2,activation='softmax', name='out_bool')(drop_4)
    
    model = keras.models.Model(inputs=[inp], outputs=[out_snr, out_bool])
    
    return(model)

def get_formatted_data(file_path):
    unformatted_tr_data = load_data.load_training_data(file_path)
    unformatted_tr_labels = load_data.load_training_labels(file_path)
    unformatted_te_data = load_data.load_testing_data(file_path)
    unformatted_te_labels = load_data.load_testing_labels(file_path)
    
    formatted_tr_data = unformatted_tr_data
    formatted_te_data = unformatted_te_data
    
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
    
    print(formatted_tr_labels)
    
    return(((formatted_tr_data, formatted_tr_labels), (formatted_te_data, formatted_te_labels)))

def format_data_segment(data):
    return(data)

def format_label_segment(segment):
    formatted_label = [[],[]]
    for l in segment:
        formatted_label[0].append([l[0]])
        formatted_label[1].append([1, 0] if bool(l[1]) else [0, 1])
    formatted_label = [np.array(dat) for dat in formatted_label]
    return(formatted_label)

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

def generator_fit(data_path, batch_size=32):
    with h5py.File(data_path, 'r') as data:
        tr_d = data['training']['train_data']
        tr_l = data['training']['train_labels']
        
        batch_data_shape = list(tr_d.shape)
        batch_data_shape[0] = batch_size
        batch_label_shape = list(tr_l.shape)
        batch_label_shape[0] = batch_size
        
        batch_data = np.zeros(batch_data_shape)
        batch_labels = np.zeros(batch_label_shape)
        
        idx_list = []
        
        while True:
            for i in xrange(batch_size):
                if len(idx_list) == 0:
                    idx_list = range(len(tr_d))
                idx = np.random.choice(idx_list)
                idx_list.remove(idx)
                
                batch_data[i] = tr_d[idx]
                batch_labels[i] = tr_l[idx]
            yield (format_data_segment(batch_data), format_label_segment(batch_labels))

def generator_evaluate(data_path, batch_size=32):
    with h5py.File(data_path, 'r') as data:
        te_d = data['testing']['test_data']
        te_l = data['testing']['test_labels']
        
        batch_data_shape = list(te_d.shape)
        batch_data_shape[0] = batch_size
        batch_label_shape = list(te_l.shape)
        batch_label_shape[0] = batch_size
        
        batch_data = np.zeros(batch_data_shape)
        batch_labels = np.zeros(batch_label_shape)
        
        idx_list = []
        
        while True:
            for i in xrange(batch_size):
                if len(idx_list) == 0:
                    idx_list = range(len(te_d))
                idx = np.random.choice(idx_list)
                idx_list.remove(idx)
                
                batch_data[i] = te_d[idx]
                batch_labels[i] = te_l[idx]
            yield (format_data_segment(batch_data), format_label_segment(batch_labels))

def train_model(model, data_path, net_path, epochs=None, epoch_break=10, batch_size=32):
    print("Epochs: {}\nEpoch_break={}".format(epochs, epoch_break))
    name = __file__[:-4]
    
    #Store the results of training (i.e. the loss)
    results = []
    
    #Check if epochs is None, if so try to train until the loss of the trainingsset and the one of the testingset seperate by too much
    if epochs == None:
        raise NotImplementedError('Training until overfitting occurs is not implemented in this version.')
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
        
        (train_data, train_labels), (test_data, test_labels) = load_data.load_data(data_path)
        train_labels = format_label_segment(train_labels)
        test_labels = format_label_segment(test_labels)
        
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
            #model.fit_generator(generator_fit(data_path, batch_size=batch_size), epochs=epoch_break, steps_per_epoch=np.ceil(float(load_data.get_number_training_samples(data_path)) / batch_size))
            #model.fit(train_data, train_labels, epochs=epoch_break)
            model.fit_generator(generator=training_generator, epochs=epoch_break, use_multiprocessing=True)
            
            #Iterate counter
            curr_counter += epoch_break
            print(curr_counter)
            
            #Store model after each training-cycle
            model.save(os.path.join(net_path, name + "_epoch_" + str(curr_counter) + ".hf5"))
            print("Stored net")
            
            #Evaluate the performance of the net after every cycle and store it.
            #results.append([curr_counter, model.evaluate_generator(generator_fit(data_path, batch_size=batch_size), steps=np.ceil(float(load_data.get_number_testing_samples(data_path)) / batch_size)), model.evaluate_generator(generator_evaluate(data_path, batch_size=batch_size), steps=np.ceil(float(load_data.get_number_testing_samples(data_path)) / batch_size))])
            #results.append([curr_counter, model.evaluate(train_data, train_labels), model.evaluate(test_data, test_labels)])
            results.append([curr_counter, model.evaluate_generator(generator=training_generator, use_multiprocessing=True), model.evaluate_generator(generator=training_generator, use_multiprocessing=True)])
            #print("Results: {}".format(results))
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    return(model)
