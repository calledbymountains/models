import tensorflow as tf
from object_detection.meta_architectures import ijcai_meta_arch
from nets.resnet_utils import resnet_arg_scope
from nets.resnet_v2 import resnet_v2_152

slim = tf.contrib.slim


class IJCAIResNet152FeatureExtractor(ijcai_meta_arch.IJCAIFeatureExtractor):
    def __init__(self, is_training,
                 output_stride, batch_norm_trainable=False,
                 reuse_weights=None, weight_decay=0.0,
                 output_hooks=None):
        """Constructor.

           Args:
             is_training: See base class.
             output_stride: See base class.
             batch_norm_trainable: See base class.
             reuse_weights: See base class.
             weight_decay: See base class.
             output_hooks: A list of endpoints from where the feature map
             outputs should be taken. If a string, it is converted to a list.
             If None (default), then block4 output feature map will be used.

           Raises:
             ValueError: If `first_stage_features_stride` is not 8 or 16.
           """
        if output_stride != 8 and output_stride != 16:
            raise ValueError('`output_stride` must be 8 or 16.')
        super(IJCAIResNet152FeatureExtractor, self).__init__('IJCAIResnet152Extractor',
                                                             is_training,
                                                             output_stride,
                                                             batch_norm_trainable,
                                                             reuse_weights,
                                                             weight_decay)
        if output_hooks is None:
            # Corresponds to no fusion
            output_hooks = ['resnet_v2_152/block4']

        if not isinstance(output_hooks, list):
            if isinstance(output_hooks, str):
                output_hooks = [output_hooks]
            else:
                raise ValueError('output_hooks mus be a list of strings or a '
                                 'single string.')
        self._output_hooks = output_hooks

    @property
    def output_hooks(self):
        return self._output_hooks

    def preprocess(self, resized_inputs):
        """IJCAI Detector with ResNet_v2_152 preprocessing.

         VGG style channel mean subtraction as described here:
        https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
        Note that if the number of channels is not equal to 3, the mean subtraction
        will be skipped and the original resized_inputs will be returned.

        Args:
          resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
          representing a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: A [batch, height_out, width_out, channels] float32
          tensor representing a batch of images.
        """
        if resized_inputs.shape.as_list()[3] == 3:
            channel_means = [123.68, 116.779, 103.939]
            return resized_inputs - [[channel_means]]
        else:
            return resized_inputs

    def _extract_anchor_features(self, preprocessed_inputs, scope):
        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                             'tensor of shape %s' % preprocessed_inputs.get_shape())

        with slim.arg_scope(resnet_arg_scope(
                weight_decay=self._weight_decay)):
            # Forces is_training to False to disable batch norm update.
            with slim.arg_scope([slim.batch_norm],
                                is_training=self._train_batch_norm):
                with tf.variable_scope('ResNet152',
                                       reuse=self._reuse_weights) as scope:
                    _, endpoints = resnet_v2_152(inputs=preprocessed_inputs,
                                                 num_classes=None,
                                                 output_stride=self.output_stride,
                                                 is_training=self.is_training)
                    endpoint_names = list(endpoints.keys())
                    all_hooks_present = set(self.output_hooks).issubset(endpoint_names)
                    if not all_hooks_present:
                        raise ValueError('One or more of ourput_hooks was not found in the '
                                         'endpoints of resnet_v2_152.')

                    hooked_feature_maps = [endpoints[x] for x in self.output_hooks]
                    try:
                        concatenated_feature_map = tf.concat(hooked_feature_maps,
                                                             axis=0)
                    except:
                        raise ValueError('All the feature maps were not found to be of the same size.')

                    return concatenated_feature_map
