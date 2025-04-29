import csv
import gc
import os
import tensorflow
from tensorflow import keras
from Util.custom_layers import (CustomPadLayer, CustomCropLayer,
                                                              CustomDropDimLayer, CustomExpandLayer, CustomCastLayer)

f = open('comet_model.csv', 'a+', encoding='utf-8', newline='')
csv_writer = csv.writer(f)


def no_activation(x):
    return x
objects = {'no_activation': no_activation, 'leakyrelu': keras.layers.LeakyReLU,
           'CustomPadLayer': CustomPadLayer, 'CustomCropLayer': CustomCropLayer,
           'CustomDropDimLayer': CustomDropDimLayer, 'CustomExpandLayer': CustomExpandLayer,
           'CustomCastLayer': CustomCastLayer}


model_dict_in = {'Conv1D':'conv2D',
                 'Conv2D':'conv2D',
                 'SeparableConv2D':'separable_conv2D',
                 'SeparableConv1D':'separable_conv2D',
                 'MaxPooling2D':'max_pooling2D',
                 'GlobalMaxPooling2D':'max_pooling2D',
                 'GlobalAveragePooling1D':'average_pooling2D',
                 'GlobalAveragePooling2D':'average_pooling2D',
                 'AveragePooling2D':'average_pooling2D',
                 'AveragePooling1D':'average_pooling2D',
                 'AveragePooling3D':'average_pooling2D',
                 'leakyrelu':'leakyReLU',
                 'ELU':'ELU',
                 'ThresholdedReLU':'ReLU',
                 'ReLU':'ReLU',
                 'PReLU':'PReLU',
                 'LeakyReLU':'leakyReLU',
                 'DepthwiseConv2D':'depthwise_conv2D',
                 'BatchNormalization':'BatchNorm',
                 'Average':'average_pooling2D',
                 'Conv2DTranspose':'conv2D_transpose',
                 'Conv3DTranspose':'conv2D_transpose',
                 'LayerNormalization':'BatchNorm'
                 }

model_dict_out = {
                  'Activation':'identity',
                  'Add':'identity',
                  'Dense':'identity',
                  'Dropout':'identity',
                  'Flatten':'identity',
                  'InputLayer':'identity',
                  'RepeatVector':'identity',
                  'Reshape':'identity',
                  'ZeroPadding2D':'identity',
                  'LSTM':'identity',
                  'Concatenate':'identity',
                  'Softmax':'identity',
                  'no_activation':'identity',
                  'leakyrelu':'identity',
                  'CustomPadLayer':'identity',
                  'CustomCropLayer':'identity',
                  'CustomDropDimLayer':'identity',
                  'CustomExpandLayer':'identity',
                  'CustomCastLayer':'identity',
                  'Dot':'identity',
                  'TimeDistributed':'identity',
                  'AlphaDropout':'identity',
                  'SpatialDropout2D':'identity',
                  'ZeroPadding1D':'identity',
                  'Subtract':'identity',
                  'Cropping2D':'identity',
                  'LocallyConnected2D':'identity',
                  'LocallyConnected1D':'identity',
                  'GaussianDropout':'identity',
                  'GaussianNoise':'identity',
                  'UpSampling1D':'identity',
                  'UpSampling2D':'identity',
                  'UpSampling3D':'identity',
                  'Minimum':'identity',
                  'Maximum':'identity',
                  'Multiply':'identity',
                  'Cropping3D':'identity',
                  'Cropping1D':'identity',
                  'SpatialDropout1D':'identity',
                  'ConvLSTM1D':'identity',
                  'ActivityRegularization':'identity',
                  'Bidirectional':'identity'
                  }

path = "../resnet50"

fileIndex = 1
path_list = os.listdir(path)
for eachdir in path_list:
    print(eachdir)
    print(str(fileIndex) + "/" + str(len(path_list)))
    file_list = os.listdir(path+"/"+eachdir)
    for eachfile in file_list:
        fileIndex += 1
        if eachfile[-3:] == '.h5':
            print(eachfile)
            try:
                model = tensorflow.keras.models.load_model(path + "/" + eachdir + "/" + eachfile,custom_objects=objects, compile=False)
            except Exception as e:
             continue
            model_str = ""



            #注：conv2d后面要加上relu。
            # model.to_json()
            # with open('./save.json', 'w') as w:
            #     w.write(model.to_json())
            # print(model.name)
            # model.summary()
            # def log_model_summary(text):
            #     a = text


            layer_names = []
            layer_types = []
            final_model = ""

            if model.name[0:10] != 'sequential':
                for i in range(len(model.layers)):
                    # print("layer"+ str(i) )
                    this_layer = model.get_layer(index = i)
                    this_layer_name = this_layer.name
                    this_layer_type = this_layer.__class__.__name__
                    if this_layer_type in model_dict_in:
                        this_layer_type = model_dict_in[this_layer_type]
                    else:
                        this_layer_type = model_dict_out[this_layer_type]

                    if this_layer_name not in layer_names:
                        layer_names.append(this_layer_name)
                        layer_types.append(this_layer_type)
                    for inbound_layer, node_index, tensor_index, _ in this_layer._inbound_nodes[0].iterate_inbound():
                        result = ""
                        fromIndex = layer_names.index(inbound_layer.name)
                        toIndex = len(layer_names) - 1
                        final_model = final_model + "from:" + str(fromIndex) + ",to:" + str(toIndex) + ",operator:" + this_layer_type + " "
            else:
                for i in range(len(model.layers)):
                    # print("layer"+ str(i) )
                    this_layer = model.get_layer(index = i)
                    this_layer_name = this_layer.name
                    this_layer_type = this_layer.__class__.__name__
                    if this_layer_type in model_dict_in:
                        this_layer_type = model_dict_in[this_layer_type]
                    else:
                        this_layer_type = model_dict_out[this_layer_type]
                    layer_names.append(this_layer_name)
                    layer_types.append(this_layer_type)
                    fromIndex = len(layer_names) - 1
                    toIndex = len(layer_names)
                    final_model = final_model + "from:" + str(fromIndex) + ",to:" + str(toIndex) + ",operator:" + this_layer_type + " "
            csv_writer.writerow([final_model])
            del model
            gc.collect(generation=2)

