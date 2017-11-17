
#Defines a class that calls a weight
import tensorflow as tf
class CheckpointWeightsDict(object):

    checkpoint_path=''
    def _init_(self,path='/home/gzm/ProfessionalLearning/project/classicalNetwork/AlexNet/finetune_alexnet_with_tensorflow-master/checkpoint/model_epoch996.ckpt'):
        self.checkpoint_path = path

    def checkpoint_weights(self):
        #turn content of checkpointfile into a dict 

        reader = tf.train.NewCheckpointReader(self.checkpoint_path)
        variables = reader.get_variable_to_shape_map()
        '''
        for i in variables:
            print(i)
            print(type(i))
        '''
        fc8_w = reader.get_tensor('fc8/weights')
        fc8_b = reader.get_tensor('fc8/biases')
        fc7_w = reader.get_tensor('fc7/weights')
        fc7_b = reader.get_tensor('fc7/biases')
        fc6_w = reader.get_tensor('fc6/weights')
        fc6_b = reader.get_tensor('fc6/biases')
        conv3_w = reader.get_tensor('conv3/weights')
        conv3_b = reader.get_tensor('conv3/biases')
        conv2_w = reader.get_tensor('conv2/weights')
        conv2_b = reader.get_tensor('conv2/biases')
        conv1_w = reader.get_tensor('conv1/weights')
        conv1_b = reader.get_tensor('conv1/biases')
        conv5_w = reader.get_tensor('conv5/weights')
        conv5_b = reader.get_tensor('conv5/biases')
        conv4_w = reader.get_tensor('conv4/weights')
        conv4_b = reader.get_tensor('conv4/biases')

        #set up a list
        conv1 = [conv1_w,conv1_b]
        conv2 = [conv2_w,conv2_b]
        conv3 = [conv3_w,conv3_b]
        conv4 = [conv4_w,conv4_b]
        conv5 = [conv5_w,conv5_b]
        fc6 = [fc6_w,fc6_b]
        fc7 = [fc7_w,fc7_b]
        fc8 = [fc8_w,fc8_b]
        #set up a dict
        weights_dict = {'conv1':conv1,
                        'conv2':conv2,
                        'conv3':conv3,
                        'conv4':conv4,
                        'conv5':conv5,
                        'fc6':fc6,
                        'fc7':fc7,
                        'fc8':fc8}

        #print(weights_dict['fc8'])

        '''
        print(type(fc8_w))
        print(fc8_w.shape)
        fc7_w = reader.get_tensor('fc7/weights')
        print(fc7_w.shape)
        #set up a dict
        fc8 = {}
        fc8['fc8_w'] = fc8_w
        fc8['fc8_b'] = fc8_b
        print(type(fc8))
        print(fc8.keys())
        print(fc8['fc8_w'])
        '''
        '''
        #set up a list
        fc8 = [fc8_w,fc8_b]
        print(type(fc8))
        print(fc8[0].shape)
        '''
        # you must write the absolute path
        # weithts_dict = checkpoint_weights_dict("/home/gzm/ProfessionalLearning/
        # project/classicalNetwork/AlexNet/finetune_alexnet_with_tensorflow-master/checkpoint/model_epoch996.ckpt")
        # print(weights_dict['fc8'])

        return weights_dict

