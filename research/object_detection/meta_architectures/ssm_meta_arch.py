from abc import abstractmethod
import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.builders import box_predictor_builder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import box_predictor
from object_detection.core import losses
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils

slim = tf.contrib.slim


class SSMFeatureExtractor(object):
    def __init__(self,
                 is_training,
                 features_output_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        """
        Constructor.

        Args:
        is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
        features_output_stride: Output stride of extracted feature map.
        batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a relative large batch size
        (e.g. 8), it could be desirable to enable batch norm update.
        reuse_weights: Whether to reuse variables. Default is None.
        weight_decay: float weight decay for feature extractor (default: 0.0).
        """
        self._is_training = is_training
        self._features_output_stride = features_output_stride
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Feature-extractor specific preprocessing (minus image resizing)."""
        pass

    def extract_input_features(self, preprocessed_inputs, scope):
        """Extracts the features which are the input to the SSM detection system.
        
        This function is responsible for extracting feature maps from preprocessed
        images.  These features are used by the SSM system to
        predict detect objects.
        
        Args:
        preprocessed_inputs: A [batch, height, width, channels] float tensor
        representing a batch of images.
        scope: A scope name.
        
        Returns:
         input_feature_map: A tensor with shape [batch, height, width, depth]
        """
        with tf.variable_scope(scope, values=[preprocessed_inputs]):
            return self._extract_input_features(preprocessed_inputs, scope)


    @abstractmethod
    def _extract_input_features(self, preprocessed_inputs, scope):
        """Extracts the input features for the SSM. to be overridden."""
        pass

    def restore_from_classification_checkpoint_fn(
        self,
        feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.
        
        Args:
        feature_extractor_scope: A scope name for the3
        feature extractor.
        
        Returns:
        A dict mapping variable names (to load from a checkpoint) to variables in
        the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            for scope_name in [feature_extractor_scope]:
                if variable.op.name.startswith(scope_name):
                    var_name = variable.op.name.replace(scope_name + '/', '')
                    variables_to_restore[var_name] = variable
        return variables_to_restore

class SSMMetaArch(model.DetectionModel):
    """SSM Meta-architecture definition."""
    def __init__(self,
                 is_training,
                 num_classes,
                 image_resizer_fn,
                 features_extractor,
                 depthwise_separable_layer_scope_fn,
                 deformable_conv_layer_scope_fn,
                 semantic_attention_layer_scope_fn,
                 attention_combiner_scope_fn,
                 attention_reducer_scope_fn,
                 anchor_generator,
                 max_proposals,
                 crop_and_resize_fn,
                 initial_crop_size,
                 maxpool_kernel_size,
                 first_stage_localization_loss_weight,
                 first_stage_classification_loss_weight,
                 second_stage_localization_loss_weight,
                 second_stage_classification_loss_weight
                 ):
        """FasterRCNNMetaArch Constructor.
        
        Args:
        is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
        num_classes: Number of classes.  Note that num_classes *does not*
        include the background category, so if groundtruth labels take values
        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the
        assigned classification targets can range from {0,... K}).
        image_resizer_fn: A callable for image resizing.  This callable
        takes a rank-3 image tensor of shape [height, width, channels]
        (corresponding to a single image), an optional rank-3 instance mask
        tensor of shape [num_masks, height, width] and returns a resized rank-3
        image tensor, a resized mask tensor if one was provided in the input. In
        addition this callable must also return a 1-D tensor of the form
        [height, width, channels] containing the size of the true image, as the
        image resizer can perform zero padding. See protos/image_resizer.proto.
        feature_extractor: A SSMFeatureExtractor object.
        anchor_generator: An anchor_generator.AnchorGenerator object
        (note that currently we only support
        grid_anchor_generator.GridAnchorGenerator objects)
        #TODO : Add documentation of other arguments
        """
        super(FasterRCNNMetaArch, self).__init__(num_classes=num_classes)
        if not isinstance(anchor_generator,
                          grid_anchor_generator.GridAnchorGenerator):
            raise ValueError('anchor_generator must be of type '
                             'grid_anchor_generator.GridAnchorGenerator.')

        self._is_training = is_training,
        self._image_resizer_fn = image_resizer_fn
        self._feature_extractor = feature_extractor,
        self._depthwise_separable_layer_scope_fn = depthwise_separable_layer_scope_fn,
        self._deformable_conv_layer_scope_fn = deformable_conv_layer_scope_fn
        self._semantic_attention_layer_scope_fn = semantic_attention_layer_scope_fn
        self._attention_combiner_scope_fn = attention_combiner_scope_fn
        self._attention_reducer_scope_fn = attention_reducer_scope_fn
        self._anchor_generator = anchor_generator
        self._max_proposals = max_proposals
        self._crop_and_resize_fn = crop_and_resize_fn
        self._initial_crop_size = initial_crop_size
        self._maxpool_kernel_size = maxpool_kernel_size
        self._first_stage_localization_loss_weight = first_stage_localization_loss_weight
        self._first_stage_classification_loss_weight = first_stage_classification_loss_weight
        self._second_stage_localization_loss_weight = second_stage_localization_loss_weight
        self._second_stage_classification_loss_weight = second_stage_classification_loss_weight
        
                 



        
                 
