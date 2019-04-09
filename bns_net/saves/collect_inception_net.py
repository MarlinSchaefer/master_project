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
    NUM_DETECTORS = 2
    inp_1s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_1s')
    inp_2s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_2s')
    inp_4s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_4s')
    inp_8s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_8s')
    inp_16s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_16s')
    inp_32s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_32s')
    inp_64s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='inp_64s')
    
    batch_1_1s = keras.layers.BatchNormalization()(inp_1s)
    batch_1_2s = keras.layers.BatchNormalization()(inp_2s)
    batch_1_4s = keras.layers.BatchNormalization()(inp_4s)
    batch_1_8s = keras.layers.BatchNormalization()(inp_8s)
    batch_1_16s = keras.layers.BatchNormalization()(inp_16s)
    batch_1_32s = keras.layers.BatchNormalization()(inp_32s)
    batch_1_64s = keras.layers.BatchNormalization()(inp_64s)
    
    DROPOUT_RATE = 0.25
    
    dropout_1_1s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_1s)
    dropout_1_2s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_2s)
    dropout_1_4s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_4s)
    dropout_1_8s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_8s)
    dropout_1_16s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_16s)
    dropout_1_32s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_32s)
    dropout_1_64s = keras.layers.Dropout(DROPOUT_RATE)(batch_1_64s)
    
    inc_1_1s = incp_lay(dropout_1_1s, 32)
    inc_1_2s = incp_lay(dropout_1_2s, 32)
    inc_1_4s = incp_lay(dropout_1_4s, 32)
    inc_1_8s = incp_lay(dropout_1_8s, 32)
    inc_1_16s = incp_lay(dropout_1_16s, 32)
    inc_1_32s = incp_lay(dropout_1_32s, 32)
    inc_1_64s = incp_lay(dropout_1_64s, 32)
    
    batch_2_1s = keras.layers.BatchNormalization()(inc_1_1s)
    batch_2_2s = keras.layers.BatchNormalization()(inc_1_2s)
    batch_2_4s = keras.layers.BatchNormalization()(inc_1_4s)
    batch_2_8s = keras.layers.BatchNormalization()(inc_1_8s)
    batch_2_16s = keras.layers.BatchNormalization()(inc_1_16s)
    batch_2_32s = keras.layers.BatchNormalization()(inc_1_32s)
    batch_2_64s = keras.layers.BatchNormalization()(inc_1_64s)

    inc_2_1s = incp_lay(batch_2_1s, 32)
    inc_2_2s = incp_lay(batch_2_2s, 32)
    inc_2_4s = incp_lay(batch_2_4s, 32)
    inc_2_8s = incp_lay(batch_2_8s, 32)
    inc_2_16s = incp_lay(batch_2_16s, 32)
    inc_2_32s = incp_lay(batch_2_32s, 32)
    inc_2_64s = incp_lay(batch_2_64s, 32)
    
    pool_1_1s = keras.layers.MaxPooling1D(2)(inc_2_1s)
    pool_1_2s = keras.layers.MaxPooling1D(2)(inc_2_2s)
    pool_1_4s = keras.layers.MaxPooling1D(2)(inc_2_4s)
    pool_1_8s = keras.layers.MaxPooling1D(2)(inc_2_8s)
    pool_1_16s = keras.layers.MaxPooling1D(2)(inc_2_16s)
    pool_1_32s = keras.layers.MaxPooling1D(2)(inc_2_32s)
    pool_1_64s = keras.layers.MaxPooling1D(2)(inc_2_64s)
    
    batch_3_1s = keras.layers.BatchNormalization()(pool_1_1s)
    batch_3_2s = keras.layers.BatchNormalization()(pool_1_2s)
    batch_3_4s = keras.layers.BatchNormalization()(pool_1_4s)
    batch_3_8s = keras.layers.BatchNormalization()(pool_1_8s)
    batch_3_16s = keras.layers.BatchNormalization()(pool_1_16s)
    batch_3_32s = keras.layers.BatchNormalization()(pool_1_32s)
    batch_3_64s = keras.layers.BatchNormalization()(pool_1_64s)
    
    inc_3_1s = incp_lay(batch_3_1s, 32)
    inc_3_2s = incp_lay(batch_3_2s, 32)
    inc_3_4s = incp_lay(batch_3_4s, 32)
    inc_3_8s = incp_lay(batch_3_8s, 32)
    inc_3_16s = incp_lay(batch_3_16s, 32)
    inc_3_32s = incp_lay(batch_3_32s, 32)
    inc_3_64s = incp_lay(batch_3_64s, 32)
    
    batch_4_1s = keras.layers.BatchNormalization()(inc_3_1s)
    batch_4_2s = keras.layers.BatchNormalization()(inc_3_2s)
    batch_4_4s = keras.layers.BatchNormalization()(inc_3_4s)
    batch_4_8s = keras.layers.BatchNormalization()(inc_3_8s)
    batch_4_16s = keras.layers.BatchNormalization()(inc_3_16s)
    batch_4_32s = keras.layers.BatchNormalization()(inc_3_32s)
    batch_4_64s = keras.layers.BatchNormalization()(inc_3_64s)
    
    inc_4_1s = incp_lay(batch_4_1s, 32)
    inc_4_2s = incp_lay(batch_4_2s, 32)
    inc_4_4s = incp_lay(batch_4_4s, 32)
    inc_4_8s = incp_lay(batch_4_8s, 32)
    inc_4_16s = incp_lay(batch_4_16s, 32)
    inc_4_32s = incp_lay(batch_4_32s, 32)
    inc_4_64s = incp_lay(batch_4_64s, 32)
    
    batch_4_1s = keras.layers.BatchNormalization()(inc_4_1s)
    batch_4_2s = keras.layers.BatchNormalization()(inc_4_2s)
    batch_4_4s = keras.layers.BatchNormalization()(inc_4_4s)
    batch_4_8s = keras.layers.BatchNormalization()(inc_4_8s)
    batch_4_16s = keras.layers.BatchNormalization()(inc_4_16s)
    batch_4_32s = keras.layers.BatchNormalization()(inc_4_32s)
    batch_4_64s = keras.layers.BatchNormalization()(inc_4_64s)
    
    inc_5_16s = incp_lay(batch_4_16s, 32)
    inc_5_32s = incp_lay(batch_4_32s, 32)
    
    batch_5_16s = keras.layers.BatchNormalization()(inc_5_16s)
    batch_5_32s = keras.layers.BatchNormalization()(inc_5_32s)
    
    inc_6_16s = incp_lay(batch_5_16s, 32)
    inc_6_32s = incp_lay(batch_5_32s, 32)
    
    batch_6_16s = keras.layers.BatchNormalization()(inc_6_16s)
    batch_6_32s = keras.layers.BatchNormalization()(inc_6_32s)
    
    pool_2_1s = keras.layers.MaxPooling1D(4)(batch_4_1s)
    pool_2_2s = keras.layers.MaxPooling1D(4)(batch_4_2s)
    pool_2_4s = keras.layers.MaxPooling1D(4)(batch_4_4s)
    pool_2_8s = keras.layers.MaxPooling1D(4)(batch_4_8s)
    pool_2_16s = keras.layers.MaxPooling1D(4)(batch_6_16s)
    pool_2_32s = keras.layers.MaxPooling1D(4)(batch_6_32s)
    pool_2_64s = keras.layers.MaxPooling1D(4)(batch_4_64s)
    
    dim_red_1_1s = keras.layers.Conv1D(16, 1)(pool_2_1s)
    dim_red_1_2s = keras.layers.Conv1D(16, 1)(pool_2_2s)
    dim_red_1_4s = keras.layers.Conv1D(16, 1)(pool_2_4s)
    dim_red_1_8s = keras.layers.Conv1D(16, 1)(pool_2_8s)
    dim_red_1_16s = keras.layers.Conv1D(16, 1)(pool_2_16s)
    dim_red_1_32s = keras.layers.Conv1D(16, 1)(pool_2_32s)
    dim_red_1_64s = keras.layers.Conv1D(16, 1)(pool_2_64s)
    
    flatten_1_1s = keras.layers.Flatten()(dim_red_1_1s)
    flatten_1_2s = keras.layers.Flatten()(dim_red_1_2s)
    flatten_1_4s = keras.layers.Flatten()(dim_red_1_4s)
    flatten_1_8s = keras.layers.Flatten()(dim_red_1_8s)
    flatten_1_16s = keras.layers.Flatten()(dim_red_1_16s)
    flatten_1_32s = keras.layers.Flatten()(dim_red_1_32s)
    flatten_1_64s = keras.layers.Flatten()(dim_red_1_64s)
    
    combined = keras.layers.concatenate([flatten_1_1s, flatten_1_2s, flatten_1_4s, flatten_1_8s, flatten_1_16s, flatten_1_32s, flatten_1_64s])
    
    #batch_4 = keras.layers.BatchNormalization()(combined)
    
    dense_1 = keras.layers.Dense(2)(combined)
    dense_2 = keras.layers.Dense(1, activation='relu', name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(3)(combined)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    model = keras.models.Model(inputs=[inp_1s, inp_2s, inp_4s, inp_8s, inp_16s, inp_32s, inp_64s], outputs=[dense_2, dense_4])
    
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
    ret = [np.zeros((2, len(data[0]), len(data))) for i in range(7)]
    ret[0][0] = tmp[0]
    ret[0][1] = tmp[7]
    
    ret[1][0] = tmp[1]
    ret[1][1] = tmp[8]
    
    ret[2][0] = tmp[2]
    ret[2][1] = tmp[9]
    
    ret[3][0] = tmp[3]
    ret[3][1] = tmp[10]
    
    ret[4][0] = tmp[4]
    ret[4][1] = tmp[11]
    
    ret[5][0] = tmp[5]
    ret[5][1] = tmp[12]
    
    ret[6][0] = tmp[6]
    ret[6][1] = tmp[13]
    
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
            model.fit_generator(generator=training_generator, epochs=epoch_break, max_q_size=1)
            
            #Iterate counter
            curr_counter += epoch_break
            print(curr_counter)
            
            #Store model after each training-cycle
            model.save(os.path.join(net_path, name + "_epoch_" + str(curr_counter) + ".hf5"))
            print("Stored net")
            
            #Evaluate the performance of the net after every cycle and store it.
            results.append([curr_counter, model.evaluate_generator(generator=training_generator, max_q_size=1), model.evaluate_generator(generator=testing_generator, max_q_size=1)])
            #print("Results: {}".format(results))
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    return(model)
