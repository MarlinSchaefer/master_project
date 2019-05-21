import keras
import numpy as np
import json
import os
import load_data
import generator as g
from data_object import DataSet

def incp_lay(x, filter_num):
    active_filter_sizes = (4, 8, 16)
    l = keras.layers.Conv1D(3 * filter_num, active_filter_sizes[0], padding='same', activation='relu')(x)
    lm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    lm_2 = keras.layers.Conv1D(2 * filter_num, active_filter_sizes[1], padding='same', activation='relu')(lm_1)
    rm_1 = keras.layers.Conv1D(int(round(filter_num / 2.0)), 1, activation='relu')(x)
    rm_2 = keras.layers.Conv1D(filter_num, active_filter_sizes[2], padding='same', activation='relu')(rm_1)
    r_1 = keras.layers.MaxPooling1D(4, strides=1, padding='same')(x)
    r_2 = keras.layers.Conv1D(int(round(filter_num)), 1, activation='relu')(r_1)
    
    outp = keras.layers.concatenate([l, lm_2, rm_2, r_2])
    
    return(outp)

def stack(x, NUM_DETECTORS, DROPOUT_RATE):
    batch_1 = keras.layers.BatchNormalization()(x)
    dropout_1 = keras.layers.Dropout(DROPOUT_RATE)(batch_1)
    conv_1 = keras.layers.Conv1D(64, 32)(dropout_1)
    bn_conv_1 = keras.layers.BatchNormalization()(conv_1)
    act_conv_1 = keras.layers.Activation('relu')(bn_conv_1)
    pool_conv_1 = keras.layers.MaxPooling1D(4)(act_conv_1)
    conv_2 = keras.layers.Conv1D(128, 16)(pool_conv_1)
    bn_conv_2 = keras.layers.BatchNormalization()(conv_2)
    act_conv_2 = keras.layers.Activation('relu')(bn_conv_2)
    inc_1 = incp_lay(act_conv_2, 32)
    batch_2 = keras.layers.BatchNormalization()(inc_1)
    inc_2 = incp_lay(batch_2, 32)
    pool_1 = keras.layers.MaxPooling1D(2)(inc_2)
    batch_3 = keras.layers.BatchNormalization()(pool_1)
    inc_3 = incp_lay(batch_3, 32)
    batch_4 = keras.layers.BatchNormalization()(inc_3)
    return(batch_4)

def get_model():
    NUM_DETECTORS = 2
    DROPOUT_RATE = 0.25
    
    inp_1s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_1s')
    inp_2s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_2s')
    inp_4s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_4s')
    inp_8s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_8s')
    inp_16s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_16s')
    inp_32s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_32s')
    inp_64s = keras.layers.Input(shape=(4096, NUM_DETECTORS, ), name='Input_64s')
    
    stack_1s = stack(inp_1s, NUM_DETECTORS, DROPOUT_RATE)
    stack_2s = stack(inp_2s, NUM_DETECTORS, DROPOUT_RATE)
    stack_4s = stack(inp_4s, NUM_DETECTORS, DROPOUT_RATE)
    stack_8s = stack(inp_8s, NUM_DETECTORS, DROPOUT_RATE)
    stack_16s = stack(inp_16s, NUM_DETECTORS, DROPOUT_RATE)
    stack_32s = stack(inp_32s, NUM_DETECTORS, DROPOUT_RATE)
    stack_64s = stack(inp_64s, NUM_DETECTORS, DROPOUT_RATE)
    
    #COMBINED OUTPUT
    combined = keras.layers.concatenate([stack_1s, stack_2s, stack_4s, stack_8s, stack_16s, stack_32s, stack_64s])
    
    inc_4 = incp_lay(combined, 32)
    batch_4 = keras.layers.BatchNormalization()(inc_4)
    pool_2 = keras.layers.MaxPooling1D(4)(batch_4)
    inc_5 = incp_lay(pool_2, 32)
    batch_5 = keras.layers.BatchNormalization()(inc_5)
    dim_red = keras.layers.Conv1D(16, 1)(batch_5)
    
    #FLAT LAYERS
    flatten = keras.layers.Flatten()(dim_red)
    
    dense_1 = keras.layers.Dense(2)(flatten)
    dense_2 = keras.layers.Dense(1, activation='relu', name='Out_SNR')(dense_1)
    
    dense_3 = keras.layers.Dense(3)(flatten)
    dense_4 = keras.layers.Dense(2, activation='softmax', name='Out_Bool')(dense_3)
    
    model = keras.models.Model(inputs=[inp_1s, inp_2s, inp_4s, inp_8s, inp_16s, inp_32s, inp_64s], outputs=[dense_2, dense_4])
    #model = keras.models.Model(inputs=[inp_2s, inp_8s, inp_32s], outputs=[dense_2, dense_4])
    
    return(model)

def compile_model(model):
    model.compile(loss={'Out_SNR': 'mean_squared_error', 'Out_Bool': 'categorical_crossentropy'}, loss_weights={'Out_SNR': 1.0, 'Out_Bool': 0.5}, optimizer='adam', metrics={'Out_SNR': 'mape', 'Out_Bool': 'accuracy'})

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

def get_data_obj(file_path):
    class CustomDataSet(DataSet):
        def format_data_segment(self, data):
            if data == []:
                return(data)
            tmp = np.array(data).transpose((2, 1, 0))
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
        
        def format_label_segment(self, data):
            ret = [[], []]
            for l in data:
                ret[0].append([l[0]])
                ret[1].append([1, 0] if bool(l[1]) else [0, 1])
            ret = [np.array(dat) for dat in ret]
            return(ret)
        
        def join_formatted(self, t, s, part1, part2):
            if part1 == None:
                part1 = []
            if part2 == None:
                part2 = []
            if self.loaded_data[t][s] == None:
                self.loaded_data[t][s] = []
            
            if s in ['train_data', 'test_data']:
                max_ind = max([len(part1), len(self.loaded_data[t][s]), len(part2)])
                if max_ind == 0:
                    self.loaded_data[t][s] = None
                    return
                
                if part1 == []:
                    part1 = [[] for i in range(max_ind)]
                if part2 == []:
                    part2 = [[] for i in range(max_ind)]
                if self.loaded_data[t][s] == []:
                    self.loaded_data[t][s] = [[] for i in range(max_ind)]
                
                self.loaded_data[t][s] = [np.array(list(part1[i]) + list(self.loaded_data[t][s][i]) + list(part2[i])) for i in range(max_ind)]
                
                if self.loaded_data[t][s] == [[] for i in range(max_ind)]:
                    self.loaded_data[t][s] = None
                return
            elif s in ['train_labels', 'test_labels']:
                if part1 == None or part1 == []:
                    part1 = [[], []]
                if part2 == None or part2 == []:
                    part2 = [[], []]
                if self.loaded_data[t][s] == None or self.loaded_data[t][s] == []:
                    self.loaded_data[t][s] = [[], []]
                
                self.loaded_data[t][s] = [np.array(list(part1[0]) + list(self.loaded_data[t][s][0]) + list(part2[0])), np.array(list(part1[1]) + list(self.loaded_data[t][s][1]) + list(part2[1]))]
                
                if self.loaded_data[t][s] == [[], []]:
                    self.loaded_data[t][s] = None
                
                return
            elif s in ['train_snr_calculated', 'test_snr_calculated']:
                if part1 == None:
                    part1 = []
                if part2 == None:
                    part2 = []
                if self.loaded_data[t][s] == None:
                    self.loaded_data[t][s] = part1 + part2
                else:
                    self.loaded_data[t][s] = part1 + self.loaded_data[t][s] + part2
                
                if self.loaded_data[t][s] == []:
                    self.loaded_data[t][s] = None
                
                return
    
    return(CustomDataSet(file_path))

def get_generator():
    return(g.DataGeneratorMultInput)

def get_model_memory_usage(batch_size, model):
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def train_model(model, dobj, net_path, epochs=None, epoch_break=10, batch_size=32):
    print("Epochs: {}\nEpoch_break={}".format(epochs, epoch_break))
    print("Net path: {}".format(net_path))
    name = os.path.basename(__file__)[:-4]
    
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
        
        print("Expected memory_size: {}".format(get_model_memory_usage(batch_size, model)))
        
        training_generator = get_generator()
        testing_generator = get_generator()
        
        training_generator = training_generator(dobj.loaded_train_data, dobj.loaded_train_labels, batch_size=batch_size)
        testing_generator = testing_generator(dobj.loaded_test_data, dobj.loaded_test_labels, batch_size=batch_size)
        
        for i in range(ran):
            print("ran: {}\ni: {}".format(ran, i))
            #If epochs were not an integer multiple of epoch_break, the last training cycle has to be smaller
            if i == int(epochs / epoch_break):
                epoch_break = epochs - (ran - 1) * epoch_break
                #Handle the exception of epochs < epoch_break
                if epoch_break < 0:
                    epoch_break += epoch_break
            
            q_size = 2
            
            #Fit data to model            
            model.fit_generator(generator=training_generator, epochs=epoch_break, max_q_size=q_size)
            
            #Iterate counter
            curr_counter += epoch_break
            print(curr_counter)
            
            #Store model after each training-cycle
            print("Net path before saving: {}".format(net_path))
            tmp_name = str(name + "_epoch_" + str(curr_counter) + ".hf5")
            tmp_path = os.path.join(net_path, tmp_name)
            print("Trying to save at: {}".format(tmp_path))
            model.save(os.path.join(net_path, tmp_name))
            print("Stored net at: {}".format(os.path.join(net_path, tmp_name)))
            
            #Evaluate the performance of the net after every cycle and store it.
            results.append([curr_counter, model.evaluate_generator(generator=training_generator, max_q_size=q_size), model.evaluate_generator(generator=testing_generator, max_q_size=q_size)])
            #print("Results: {}".format(results))
    
    #Save the results to a file.
    with open(os.path.join(net_path, name + '_results.json'), "w+") as FILE:
        json.dump(results, FILE, indent=4)
    
    return(model)
