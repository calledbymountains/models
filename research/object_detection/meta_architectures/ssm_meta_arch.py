from abc import abstractmethod
import tensorflow as tf
import tensorlayer as tl
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
                 semantic_attention_layer_scope_fn,
                 attention_combiner_scope_fn,
                 attention_reducer_scope_fn,
                 anchor_generator,
                 max_centers,
                 selection_threshold,
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
        self._semantic_attention_layer_scope_fn = semantic_attention_layer_scope_fn
        self._attention_combiner_scope_fn = attention_combiner_scope_fn
        self._attention_reducer_scope_fn = attention_reducer_scope_fn
        self._anchor_generator = anchor_generator
        self._max_centers = max_centers
        self._selection_threshold = selection_threshold
        self._crop_and_resize_fn = crop_and_resize_fn
        self._initial_crop_size = initial_crop_size
        self._maxpool_kernel_size = maxpool_kernel_size
        self._first_stage_localization_loss_weight = first_stage_localization_loss_weight
        self._first_stage_classification_loss_weight = first_stage_classification_loss_weight
        self._second_stage_localization_loss_weight = second_stage_localization_loss_weight
        self._second_stage_classification_loss_weight = second_stage_classification_loss_weight

    def preprocess(self, inputs):
        """Feature-extractor specific preprocessing.
        
        See base class.
        
        For SSM, we perform image resizing in the base class --- each
        class subclassing FasterRCNNMetaArch is responsible for any additional
        preprocessing (e.g., scaling pixel values to be in [-1, 1]).
        
        Args:
        inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.
        
        Returns:
        preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
        true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
        Raises:
        ValueError: if inputs tensor does not have type tf.float32
        """
    if inputs.dtype is not tf.float32:
      raise ValueError('`preprocess` expects a tf.float32 tensor')
    with tf.name_scope('Preprocessor'):
      outputs = shape_utils.static_or_dynamic_map_fn(
          self._image_resizer_fn,
          elems=inputs,
          dtype=[tf.float32, tf.int32],
          parallel_iterations=self._parallel_iterations)
      resized_inputs = outputs[0]
      true_image_shapes = outputs[1]
      return (self._feature_extractor.preprocess(resized_inputs),
              true_image_shapes)

    def _compute_clip_window(self, image_shapes):
    """Computes clip window for non max suppression based on image shapes.

    This function assumes that the clip window's left top corner is at (0, 0).

    Args:
      image_shapes: A 2-D int32 tensor of shape [batch_size, 3] containing
      shapes of images in the batch. Each row represents [height, width,
      channels] of an image.

    Returns:
      A 2-D float32 tensor of shape [batch_size, 4] containing the clip window
      for each image in the form [ymin, xmin, ymax, xmax].
    """
    clip_heights = image_shapes[:, 0]
    clip_widths = image_shapes[:, 1]
    clip_window = tf.to_float(tf.stack([tf.zeros_like(clip_heights),
                                        tf.zeros_like(clip_heights),
                                        clip_heights, clip_widths], axis=1))
    return clip_window

    @property
    def feature_extractor_scope(self):
        return 'FeatureExtractor'

    def predict(self, preprocessed_inputs, true_image_shapes):
        """Predicts unpostprocessed tensors from input tensor.
        
        This function takes an input batch of images and runs it through the
        forward pass of the network to yield "raw" un-postprocessed predictions.
        If `number_of_stages` is 1, this function only returns first stage
        RPN predictions (un-postprocessed).  Otherwise it returns both
        first stage RPN predictions as well as second stage box classifier
        predictions.

        Other remarks:
        + Anchor pruning vs. clipping: following the recommendation of the Faster
        R-CNN paper, we prune anchors that venture outside the image window at
        training time and clip anchors to the image window at inference time.
        + Proposal padding: as described at the top of the file, proposals are
        padded to self._max_num_proposals and flattened so that proposals from all
        images within the input batch are arranged along the same batch dimension.
        
        Args:
        preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
        true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
        
        Returns:
        prediction_dict: a dictionary holding "raw" prediction tensors:
        1) rpn_box_predictor_features: A 4-D float32 tensor with shape
        [batch_size, height, width, depth] to be used for predicting proposal
        boxes and corresponding objectness scores.
        2) rpn_features_to_crop: A 4-D float32 tensor with shape
        [batch_size, height, width, depth] representing image features to crop
        using the proposal boxes predicted by the RPN.
        3) image_shape: a 1-D tensor of shape [4] representing the input
        image shape.
        4) rpn_box_encodings:  3-D float tensor of shape
        [batch_size, num_anchors, self._box_coder.code_size] containing
        predicted boxes.
        5) rpn_objectness_predictions_with_background: 3-D float tensor of shape
        [batch_size, num_anchors, 2] containing class
        predictions (logits) for each of the anchors.  Note that this
        tensor *includes* background class predictions (at class index 0).
        6) anchors: A 2-D tensor of shape [num_anchors, 4] representing anchors
        for the first stage RPN (in absolute coordinates).  Note that
        `num_anchors` can differ depending on whether the model is created in
        training or inference mode.
        
        (and if number_of_stages > 1):
        7) refined_box_encodings: a 3-D tensor with shape
        [total_num_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings, where
        total_num_proposals=batch_size*self._max_num_proposals. If using
        a shared box across classes the shape will instead be
        [total_num_proposals, 1, self._box_coder.code_size].
        8) class_predictions_with_background: a 3-D tensor with shape
        [total_num_proposals, num_classes + 1] containing class
        predictions (logits) for each of the anchors, where
        total_num_proposals=batch_size*self._max_num_proposals.
        Note that this tensor *includes* background class predictions
        (at class index 0).
        9) num_proposals: An int32 tensor of shape [batch_size] representing the
        number of proposals generated by the RPN.  `num_proposals` allows us
        to keep track of which entries are to be treated as zero paddings and
        which are not since we always pad the number of proposals to be
        `self.max_num_proposals` for each image.
        10) proposal_boxes: A float32 tensor of shape
        [batch_size, self.max_num_proposals, 4] representing
        decoded proposal bounding boxes in absolute coordinates.
        11) mask_predictions: (optional) a 4-D tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask predictions.
        
        Raises:
        ValueError: If `predict` is called before `preprocess`.
        """
        image_shape = tf.shape(preprocessed_inputs)
        feature_extractor_output_map = self._feature_extractor.extract_input_features(
            preprocessed_inputs,
            scope=self.feature_extractor_scope
            )
        depthwise_separable_output = self.build_depthwise_separable_layer(feature_extractor_output_map, self._is_training)
        deformable_feature_output = self.build_deformable_conv_layer(depthwise_separable_output, self._is_training)
        semantic_attention_feature_output = self.build_semantic_attention_layer(deformable_feature_output, self._is_training)
        attention_combiner_feature_output = self.build_attention_combiner_layer(semantic_attention_feature_output, self._is_training)
        attention_confidence_output = tf.nn.softmax(attention_combiner_feature_output,
                                                    axis=3)
        pedestrian_confidence_output = attention_confidence_output[:,:,:,1]
        pedestrian_selector_feature = tf.concat([pedestrian_confidence_output,
                                                 deformable_feature_output],
                                                axis=3)
        pedestrian_reducer_feature_output = self.build_attention_reducer_layer(pedestrian_selector_feature,
                                                                               is_training)
        pedestrian_selection_feature_map = tf.layers.conv2d(pedestrian_reducer_feature_output,
                                                            filters=1,
                                                            kernel_size=3,
                                                            padding='same',
                                                            activation=tf.nn.relu,
                                                            is_training=self._is_training
                                                            )

        valid_pedestrian_locations = tf.where(tf.greater_equal(pedestrian_selection_feature_map, self._selection_threshold))
        num_anchors_per_location = self._anchor_generator.num_anchors_per_location()
        if len(num_anchors_per_location) != 1:
            raise RuntimeError('anchor_generator is expected to generate anchors '
                               'corresponding to a single feature map.')
        anchors = self._anchor_generator.generate([pedestrian_selection_feature_map.shape[1],
                                                         pedestrian_selection_feature_map.shape[2])])

        anchors_valid_locations_unit = tf.reshape(valid_pedestrian_locations, [image_shape[0], -1,1])
        anchors_valid_locations = tf.tile(anchors_valid_locations_unit, [1, num_anchors_per_location])
        anchors_valid_locations = tf.reshape(anchors_valid_locations, [-1,1])
        selected_anchors = tf.gather(anchors, anchors_valid_locations)
        pass
    

    def convert_to_minibatch(self, valid_pedestrian_locations, anchors):
        pass


    def predict_coarse_stage(self, feature_map_to_crop, selected_anchors, image_shape, true_image_shapes):
        """Predicts the coarse stage of classification and prediction."""
        image_shape_2d = self._image_batch_shape_2d(image_shape)
        pass


    def _image_batch_shape_2d(self, image_batch_shape_1d):
        """Takes a 1-D image batch shape tensor and converts it to a 2-D tensor.
        
        Example:
        If 1-D image batch shape tensor is [2, 300, 300, 3]. The corresponding 2-D
        image batch tensor would be [[300, 300, 3], [300, 300, 3]]

        Args:
        image_batch_shape_1d: 1-D tensor of the form [batch_size, height,
        width, channels].
        
        Returns:
        image_batch_shape_2d: 2-D tensor of shape [batch_size, 3] were each row is
        of the form [height, width, channels].
        """
        return tf.tile(tf.expand_dims(image_batch_shape_1d[1:], 0),
                       [image_batch_shape_1d[0], 1])

    def build_depthwise_separable_layer(self, feature_extractor_output_map, is_training):
        with slim.arg_scope(self._depthwise_separable_layer_scope_fn):
            depthwise_feature_out = tf.layers.separable_conv2d(
                inputs=feature_extractor_output_map,
                filters=512,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                depthwise_initializer=tf.initializers.truncated_normal(
                    mean=0.0,
                    stddev=0.01),
                pointwise_initializer=tf.initializers.truncated_normal(
                    mean=0.0,
                    stddev=0.01),
                depthwise_regularizer=tf.contrib.layers.l2_regularizer(
                    scale=0.0005
                    ),
                pointwise_regularizer=tf.contrib.layers.l2_regularizer(
                    scale=0.0005
                    ),
                trainable=is_training
                )
        return depthwise_feature_out

    def build_deformable_conv_layer(self, depthwise_feature_ouutput, is_training):
        offset = tl.layers.Conv2d(depthwise_feature_output, 18, (3,3), (1,1), act=None,
                                  padding='same')
        deformable_feature_output = tl.layers.DeformableConv2d(depthwise_feature_output,
                                                               offset,
                                                               512,
                                                               act=tf.nn.relu
        )
        return deformable_feature_output

    def build_semantic_attention_layer(self, depthwise_feature_output, is_training):
        atrous_rates = list(range(len(self._semantic_attention_layer_scope_fn)))
        atrous_rates+=1
        semantic_attention_outputs = []
        for attention_layer_scope_fn, atrous_rate in zip(self._semantic_attention_layer_scope_fn, atrous_rates):
            with slim.arg_scope(attention_layer_scope_fn):
                output = tf.layers.conv2d(depthwise_feature_output,
                                          filters=64,
                                          kernel_size=3,
                                          padding='same',
                                          dilation_rate=atrous_rate,
                                          activation=tf.nn.relu)
                semantic_attention_outputs.append(output)
        
        semantic_attention_output = tf.concat(semantic_attention_outputs, axis=3)
        return semantic_attention_output

    def build_attention_combiner_layer(self, attention_feature_output, is_training):
        with slim.arg_scope(self._attention_combiner_scope_fn):
            attention_combiner_output = tf.layers.conv2d(attention_feature_output,
                                                         filters=self._num_classes,
                                                         kernel_size=3,
                                                         padding='same',
                                                         activation=tf.nn.relu,
                                                         trainable=is_training)
        return attention_combiner_output

    def build_attention_reducer_layer(self, pedestrian_selector_feature, is_training):
        with slim.arg_scope(self._attention_reducer_scope_fn):
            attention_reducer_output = tf.layers.conv2d(pedestrian_selector_feature,
                                                        filters=64,
                                                        kernel_size=3,
                                                        padding='same',
                                                        activation=tf.nn.relu,
                                                        trainable=is_training)
        return attention_reducer_output
        

    
                                      
            
        
        
            
            
        

        
        

        
        
                 



        
                 
