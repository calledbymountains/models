import tensorflow as tf
from object_detection.meta_architectures import ssm_meta_arch
from nets import resnet_utils
from nets import resnet_v2


slim = tf.contrib.slim


class SSMResnetV2FeatureExtractor(
        ssm_meta_arch.SSMFeatureExtractor):
    """ SSM Resnet V2 Feature extractor implementation."""

    def __init__(self,
                 architecture,
                 resnet_model,
                 is_training,
                 features_output_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.

        Args:
            architecture: Architecture name of the Resnet V2 model.
            resnet_model: Definition of the Resnet V2 model.
            is_training: See base class.
            first_stage_features_stride: See base class.
            batch_norm_trainable: See base class.
            reuse_weights: See base class.
            weight_decay: See base class.
        Raises:
            ValueError: If `features_output_stride` is not 8 or 16.
        """

        if features_output_stride not in [8,16]:
            raise ValueError('`features_output_stride` must be 8 or 16.')
        self._architecture = architecture
        self._resnet_model = resnet_model
        super(SSMResnetV2FeatureExtractor, self).__init__(
            is_training, features_output_stride, batch_norm_trainable,
            reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):
        """Faster R-CNN Resnet V1 preprocessing.
        
        VGG style channel mean subtraction as described here:
        https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
        
        Args:
        resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.
        
        Returns:
        preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
        
        """
        channel_means = [123.68, 116.779, 103.939]
        return resized_inputs - [[channel_means]]


    def _extract_input_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.
        
        Args:
        preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
        scope: A scope name.
        
        Returns:
        feature_map: A tensor with shape [batch, height, width, depth]
        
        Raises:
        InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
        """

        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                             'tensor of shape %s' % preprocessed_inputs.get_shape())
        shape_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        with tf.control_dependencies([shape_assert]):
            with slim.arg_scope(
                    resnet_utils.resnet_arg_scope(
                        batch_norm_epsilon=1e-5,
                        batch_norm_scale=True,
                        weight_decay=self._weight_decay)):
                with tf.variable_scope(
                        self._architecture, reuse=self._reuse_weights) as var_scope:
                    _, activations = self._resnet_model(
                        preprocessed_inputs,
                        num_classes=None,
                        is_training=self._train_batch_norm,
                        global_pool=False,
                        output_stride=self._features_output_stride,
                        spatial_squeeze=False,
                        scope=var_scope)

        extraction_points = ['{}/{}/block{}'.format(scope,
                                                    self._architecture,
                                                    x) for x in [2,3,4]
                             ]
        output_concatenated = tf.concat([activations[x] for x in extraction_points],
                                        axis=3)
        return output_concatenated



class SSMResnet50FeatureExtractor(SSMResnetV2FeatureExtractor):
    """SSM Resnet 50 feature extractor implementation."""
    def __init__(self,
                 is_training,
                 features_output_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.
        
        Args:
        is_training: See base class.
        features_output_stride: See base class.
        batch_norm_trainable: See base class.
        reuse_weights: See base class.
        weight_decay: See base class.
        
        Raises:
        ValueError: If `features_output_stride` is not 8 or 16,
        or if `architecture` is not supported.
        """
        
        super(SSMResnet50FeatureExtractor, self).__init__(
            'resnet_v2_50', resnet_v2.resnet_v2_50, is_training,
            features_output_stride, batch_norm_trainable,
            reuse_weights, weight_decay)

class SSMResnet152FeatureExtractor(SSMResnetV2FeatureExtractor):
    """SSM Resnet 152 feature extractor implementation."""
    def __init__(self,
                 is_training,
                 features_output_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.
        
        Args:
        is_training: See base class.
        features_output_stride: See base class.
        batch_norm_trainable: See base class.
        reuse_weights: See base class.
        weight_decay: See base class.
        
        Raises:
        ValueError: If `features_output_stride` is not 8 or 16,
        or if `architecture` is not supported.
        """
        
        super(SSMResnet152FeatureExtractor, self).__init__(
            'resnet_v2_152', resnet_v2.resnet_v2_152, is_training,
            features_output_stride, batch_norm_trainable,
            reuse_weights, weight_decay)


    
class SSMResnet101FeatureExtractor(SSMResnetV2FeatureExtractor):
    """SSM Resnet 101 feature extractor implementation."""
    def __init__(self,
                 is_training,
                 features_output_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """Constructor.
        
        Args:
        is_training: See base class.
        features_output_stride: See base class.
        batch_norm_trainable: See base class.
        reuse_weights: See base class.
        weight_decay: See base class.
        
        Raises:
        ValueError: If `features_output_stride` is not 8 or 16,
        or if `architecture` is not supported.
        """
        
        super(SSMResnet152FeatureExtractor, self).__init__(
            'resnet_v2_101', resnet_v2.resnet_v2_101, is_training,
            features_output_stride, batch_norm_trainable,
            reuse_weights, weight_decay)
        
            


