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
                 feature_extractor,
                 depthwise_separable_layer_scope_fn,
                 semantic_attention_layer_scope_fn,
                 attention_combiner_scope_fn,
                 attention_reducer_scope_fn,
                 anchor_generator,
                 max_anchors,
                 selection_threshold,
                 crop_and_resize_fn,
                 initial_crop_size,
                 maxpool_kernel_size,
                 maxpool_kernel_stride,
                 first_stage_localization_loss_weight,
                 first_stage_classification_loss_weight,
                 second_stage_localization_loss_weight,
                 second_stage_classification_loss_weight,
                 first_stage_mask_rcnn_predictor,
                 second_stage_mask_rcnn_predictor,
                 fine_stage_nms_fn,
                 fine_stage_box_score_conversion_fn,
                 coarse_stage_target_assigner,
                 fine_stage_target_assigner,
                 coarse_stage_cls_loss,
                 fine_stage_cls_loss,
                 parallel_iterations=100,
                 add_summaries=False
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
        super(SSMMetaArch, self).__init__(num_classes=num_classes)
        if not isinstance(anchor_generator,
                          grid_anchor_generator.GridAnchorGenerator):
            raise ValueError('anchor_generator must be of type '
                             'grid_anchor_generator.GridAnchorGenerator.')

        self._is_training = is_training,
        self._image_resizer_fn = image_resizer_fn
        self._feature_extractor = feature_extractor
        self._depthwise_separable_layer_scope_fn = depthwise_separable_layer_scope_fn
        self._semantic_attention_layer_scope_fn = semantic_attention_layer_scope_fn
        self._attention_combiner_scope_fn = attention_combiner_scope_fn
        self._attention_reducer_scope_fn = attention_reducer_scope_fn
        self._anchor_generator = anchor_generator
        self._max_anchors = max_anchors
        self._selection_threshold = selection_threshold
        self._crop_and_resize_fn = crop_and_resize_fn
        self._initial_crop_size = initial_crop_size
        self._maxpool_kernel_size = maxpool_kernel_size
        self._maxpool_kernel_stride = maxpool_kernel_stride,
        self._first_stage_localization_loss_weight = first_stage_localization_loss_weight
        self._first_stage_classification_loss_weight = first_stage_classification_loss_weight
        self._second_stage_localization_loss_weight = second_stage_localization_loss_weight
        self._second_stage_classification_loss_weight = second_stage_classification_loss_weight
        self._first_stage_mask_rcnn_predictor = first_stage_mask_rcnn_predictor
        self._second_stage_mask_rcnn_predictor = second_stage_mask_rcnn_predictor
        self._parallel_iterations = parallel_iterations
        self._num_anchors_per_location = None
        self._fine_stage_nms_fn = fine_stage_nms_fn
        self._fine_stage_box_score_conversion_fn = fine_stage_box_score_conversion_fn,
        self._coarse_stage_target_assigner = coarse_stage_target_assigner
        self._fine_stage_target_assigner = fine_stage_target_assigner
        self._coarse_stage_cls_loss = coarse_stage_cls_loss
        self._fine_stage_cls_loss = fine_stage_cls_loss




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
            return self._feature_extractor.preprocess(resized_inputs), true_image_shapes

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

    @property
    def first_stage_box_predictor_scope(self):
        return 'CoarseStagePredictor'

    @property
    def second_stage_box_predictor_scope(self):
        return 'FineStagePredictor'

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
        depthwise_separable_output = self.build_depthwise_separable_layer(
            feature_extractor_output_map, self._is_training)
        deformable_feature_output = self.build_deformable_conv_layer(
            depthwise_separable_output, self._is_training)

        semantic_attention_feature_output = self.build_semantic_attention_layer(
            deformable_feature_output, self._is_training)
        attention_combiner_feature_output = self.build_attention_combiner_layer(
            semantic_attention_feature_output, self._is_training)
        attention_confidence_output = tf.nn.softmax(
            attention_combiner_feature_output,
            axis=3)
        # Select all but the background class.
        pedestrian_confidence_output = attention_confidence_output[:, :, :, 1:]
        # Concatenate with the deformable convolution output feature map.
        pedestrian_selector_feature = tf.concat([pedestrian_confidence_output,
                                                 deformable_feature_output],
                                                axis=3)
        pedestrian_reducer_feature_output = self.build_attention_reducer_layer(
            pedestrian_selector_feature,
            self._is_training)
        # Reduce the output to one feature map with one channel.
        class_selection_feature_map = tf.layers.conv2d(
            pedestrian_reducer_feature_output,
            filters=1,
            kernel_size=3,
            padding='same',
            activation=tf.nn.relu,
            is_training=self._is_training
        )

        # valid_pedestrian_locations is of shape [batchsize, height, width, 1]
        valid_locations = tf.where(tf.greater_equal(class_selection_feature_map,
                                                    self._selection_threshold))
        self._num_anchors_per_location = self._anchor_generator.num_anchors_per_location()
        if len(self._num_anchors_per_location) != 1:
            raise RuntimeError(
                'anchor_generator is expected to generate anchors '
                'corresponding to a single feature map.')
        anchors = self._anchor_generator.generate(
            [class_selection_feature_map.shape[1],
             class_selection_feature_map.shape[2]])
        # Anchors are designed as a list of length one with a BoxList.
        anchors = anchors[0]
        # We make sure that we clip all the anchors to the image size.
        anchors = box_list_ops.clip_to_window(anchors, window=[0.0, 0.0,
                                                               self._image_shape[
                                                                   1],
                                                               self._image_shape[
                                                                   2]])
        # selected_anchors_minibatch [batchsize, max_anchors, 4]
        # anchor_count [batchsize]
        selected_anchors_minibatch, anchor_count = self.select_anchor_locations_over_batch(
            valid_locations, anchors)

        coarse_prediction_dict = self.predict_coarse_stage(
            deformable_feature_output,
            selected_anchors_minibatch,
            anchor_count,
            image_shape)

        # coarse_detection_dict = self._postprocess_box_classifier(
        #     coarse_prediction_dict['refined_box_encodings'],
        #     coarse_prediction_dict['class_predictions_with_background'],
        #     coarse_prediction_dict['proposal_boxes'],
        #     coarse_prediction_dict['num_proposals'],
        #     true_image_shapes
        #     )

        refined_box_encodings = coarse_prediction_dict['refined_box_encodings']
        refined_box_encodings = self._batch_decode_boxes(refined_box_encodings,
                                                         selected_anchors_minibatch)

        refined_box_encodings = tf.reshape(refined_box_encodings, [image_shape[0], -1, 4])

        fine_prediction_dict = self.predict_fine_stage(deformable_feature_output,
                                                       refined_box_encodings,
                                                       anchor_count,
                                                       image_shape)


        # fine_detection_dict = self._postprocess_box_classifier(
        #     fine_prediction_dict['refined_box_encodings'],
        #     fine_prediction_dict['class_predictions_with_background'],
        #     fine_prediction_dict['proposal_boxes'],
        #     fine_prediction_dict['num_proposals'],
        #     true_image_shapes
        #     )

        return dict(coarse=coarse_prediction_dict,
                    fine=fine_prediction_dict,
                    spatial_softmax=class_selection_feature_map,
                    attention_combiner_feature_output=tf.image.resize_images(attention_combiner_feature_output,
                                                                             [image_shape[1], image_shape[2]])
                    )


    def _compute_input_feature_maps(self, features_to_crop,
                                    proposal_boxes_normalized):
        """Crops to a set of proposals from the feature map for a batch of images.

        Helper function for self._postprocess_rpn. This function calls
        `tf.image.crop_and_resize` to create the feature map to be passed to the
        second stage box classifier for each proposal.

        Args:
        features_to_crop: A float32 tensor with shape
        [batch_size, height, width, depth]
        proposal_boxes_normalized: A float32 tensor with shape [batch_size,
        num_proposals, box_code_size] containing proposal boxes in
        normalized coordinates.

        Returns:
        A float32 tensor with shape [K, new_height, new_width, depth].
        """
        cropped_regions = self._flatten_first_two_dimensions(
            self._crop_and_resize_fn(
                features_to_crop, proposal_boxes_normalized,
                [self._initial_crop_size, self._initial_crop_size]))
        return slim.max_pool2d(
            cropped_regions,
            [self._maxpool_kernel_size, self._maxpool_kernel_size],
            stride=self._maxpool_stride)

    def select_anchor_locations_over_batch(self, valid_locations, anchors):
        """Selects the anchor center locations."""

        def select_anchor_locations_in_one_batch(args):
            valid_location = args[0]  # [height, width, 1]
            anchor_collection = tf.squeeze(args[1], axis=0)
            anchor_collection = box_list.BoxList(anchor_collection)
            valid_location = tf.reshape(valid_location, [-1, 1])
            valid_location = tf.tile(valid_location,
                                     [1, self._num_anchors_per_location])
            valid_location = tf.reshape(valid_location, [-1])
            selected_anchors = box_list_ops.boolean_mask(anchor_collection,
                                                         valid_location)
            # We normalize the selected_anchors
            selected_anchors = box_list_ops.to_normalized_coordinates(
                selected_anchors,
                height=self._image_shape[1],
                width=self._image_shape[2])
            num_anchors = selected_anchors.num_boxes()
            num_anchors = tf.cond(tf.less_equal(num_anchors, self._max_anchors),
                                  lambda: num_anchors,
                                  lambda: self._max_anchors)
            batched_selected_anchors = box_list_ops.pad_or_clip_box_list(
                selected_anchors,
                num_anchors, self._max_anchors).get()
            return batched_selected_anchors, num_anchors

        anchors = tf.tile(tf.expand_dims(anchors.get(), axis=0),
                          [self._num_anchors_per_location, 1, 1])
        selected_anchors_minibatch = tf.map_fn(
            select_anchor_locations_in_one_batch,
            (valid_locations, anchors),
            dtype=(tf.float32, tf.int32))
        return selected_anchors_minibatch

    def predict_coarse_stage(self, feature_map_to_crop, selected_anchors,
                             num_anchors,
                             image_shape):
        """Predicts the coarse stage of classification and prediction."""
        coarse_input_feature_maps = self._compute_input_feature_maps(
            feature_map_to_crop,
            selected_anchors)

        coarse_box_predictions = self._first_stage_mask_rcnn_predictor.predict(

            [coarse_input_feature_maps],
            num_predictions_per_location=[1],
            scope=self.first_stage_box_predictor_scope,
            prediction_stage=2
        )

        refined_coarse_box_encodings = tf.squeeze(
            coarse_box_predictions[box_predictor.BOX_ENCODINGS],
            axis=1, name='all_refined_box_encodings')
        coarse_class_predictions_with_background = tf.squeeze(
            coarse_box_predictions[
                box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
            axis=1, name='all_class_predictions_with_background')

        absolute_coarse_proposal_boxes = ops.normalized_to_image_coordinates(
            selected_anchors, image_shape, self._parallel_iterations)

        coarse_prediction_dict = {
            'refined_box_encodings': refined_coarse_box_encodings,
            'class_predictions_with_background':
                coarse_class_predictions_with_background,
            'num_proposals': num_anchors,
            'proposal_boxes': absolute_coarse_proposal_boxes,
            'proposal_boxes_normalized': selected_anchors,
        }
        return coarse_prediction_dict

    def predict_fine_stage(self, feature_map_to_crop, selected_anchors,
                           num_anchors,
                           image_shape):
        """Predicts the coarse stage of classification and prediction."""
        coarse_input_feature_maps = self._compute_input_feature_maps(
            feature_map_to_crop,
            selected_anchors)

        coarse_box_predictions = self._first_stage_mask_rcnn_predictor.predict(

            [coarse_input_feature_maps],
            num_predictions_per_location=[1],
            scope=self.first_stage_box_predictor_scope,
            prediction_stage=2
        )

        refined_coarse_box_encodings = tf.squeeze(
            coarse_box_predictions[box_predictor.BOX_ENCODINGS],
            axis=1, name='all_refined_box_encodings')
        coarse_class_predictions_with_background = tf.squeeze(
            coarse_box_predictions[
                box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND],
            axis=1, name='all_class_predictions_with_background')

        absolute_coarse_proposal_boxes = ops.normalized_to_image_coordinates(
            selected_anchors, image_shape, self._parallel_iterations)

        coarse_prediction_dict = {
            'refined_box_encodings': refined_coarse_box_encodings,
            'class_predictions_with_background':
                coarse_class_predictions_with_background,
            'num_proposals': num_anchors,
            'proposal_boxes': absolute_coarse_proposal_boxes,
            'proposal_boxes_normalized': selected_anchors,
        }
        return coarse_prediction_dict

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

    def build_depthwise_separable_layer(self, feature_extractor_output_map,
                                        is_training):
        with slim.arg_scope(self._depthwise_separable_layer_scope_fn()):
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

    def build_deformable_conv_layer(self, depthwise_feature_output,
                                    is_training):
        depthwise_feature_tl = tl.layers.InputLayer(depthwise_feature_output)
        offset = tl.layers.Conv2d(depthwise_feature_tl, 18, (3, 3), (1, 1),
                                  act=None,
                                  padding='same')
        deformable_feature_output = tl.layers.DeformableConv2d(
            depthwise_feature_tl,
            offset,
            512,
            act=tf.nn.relu
        )
        return deformable_feature_output

    def build_semantic_attention_layer(self, depthwise_feature_output,
                                       is_training):
        atrous_rates = list(range(len(self._semantic_attention_layer_scope_fn)))
        atrous_rates = [x + 1 for x in atrous_rates]
        semantic_attention_outputs = []
        for attention_layer_scope_fn, atrous_rate in zip(
                self._semantic_attention_layer_scope_fn, atrous_rates):
            with slim.arg_scope(attention_layer_scope_fn()):
                output = tf.layers.conv2d(depthwise_feature_output.outputs,
                                          filters=64,
                                          kernel_size=3,
                                          padding='same',
                                          dilation_rate=atrous_rate,
                                          activation=tf.nn.relu,
                                          trainable=is_training)
                semantic_attention_outputs.append(output)

        print(semantic_attention_outputs)
        semantic_attention_output = tf.concat(semantic_attention_outputs,
                                              axis=3)
        return semantic_attention_output

    def build_attention_combiner_layer(self, attention_feature_output,
                                       is_training):
        with slim.arg_scope(self._attention_combiner_scope_fn()):
            attention_combiner_output = tf.layers.conv2d(
                attention_feature_output,
                filters=self._num_classes,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                trainable=is_training)
        return attention_combiner_output

    def build_attention_reducer_layer(self, pedestrian_selector_feature,
                                      is_training):
        with slim.arg_scope(self._attention_reducer_scope_fn):
            attention_reducer_output = tf.layers.conv2d(
                pedestrian_selector_feature,
                filters=64,
                kernel_size=3,
                padding='same',
                activation=tf.nn.relu,
                trainable=is_training)
        return attention_reducer_output

    def _flatten_first_two_dimensions(self, inputs):
        """Flattens `K-d` tensor along batch dimension to be a `(K-1)-d` tensor.

        Converts `inputs` with shape [A, B, ..., depth] into a tensor of shape
        [A * B, ..., depth].

        Args:
        inputs: A float tensor with shape [A, B, ..., depth].  Note that the first
        two and last dimensions must be statically defined.
        Returns:
        A float tensor with shape [A * B, ..., depth] (where the first and last
        dimension are statically defined.
        """
        combined_shape = shape_utils.combined_static_and_dynamic_shape(inputs)
        flattened_shape = tf.stack([combined_shape[0] * combined_shape[1]] +
                                   combined_shape[2:])
        return tf.reshape(inputs, flattened_shape)

    def _postprocess_box_classifier(self,
                                    refined_box_encodings,
                                    class_predictions_with_background,
                                    proposal_boxes,
                                    num_proposals,
                                    image_shapes,
                                    mask_predictions=None):
        """Converts predictions from the second stage box classifier to detections.

        Args:
        refined_box_encodings: a 3-D float tensor with shape
        [total_num_padded_proposals, num_classes, self._box_coder.code_size]
        representing predicted (final) refined box encodings. If using a shared
        box across classes the shape will instead be
        [total_num_padded_proposals, 1, 4]
        class_predictions_with_background: a 3-D tensor float with shape
        [total_num_padded_proposals, num_classes + 1] containing class
        predictions (logits) for each of the proposals.  Note that this tensor
        *includes* background class predictions (at class index 0).
        proposal_boxes: a 3-D float tensor with shape
        [batch_size, self.max_num_proposals, 4] representing decoded proposal
        bounding boxes in absolute coordinates.
        num_proposals: a 1-D int32 tensor of shape [batch] representing the number
        of proposals predicted for each image in the batch.
        image_shapes: a 2-D int32 tensor containing shapes of input image in the
        batch.
        mask_predictions: (optional) a 4-D float tensor with shape
        [total_num_padded_proposals, num_classes, mask_height, mask_width]
        containing instance mask prediction logits.

        Returns:
        A dictionary containing:
        `detection_boxes`: [batch, max_detection, 4] in normalized co-ordinates.
        `detection_scores`: [batch, max_detections]
        `detection_classes`: [batch, max_detections]
        `num_detections`: [batch]
        `detection_masks`:
        (optional) [batch, max_detections, mask_height, mask_width]. Note
        that a pixel-wise sigmoid score converter is applied to the detection
        masks.
        """
        refined_box_encodings_batch = tf.reshape(
            refined_box_encodings,
            [-1,
             self.max_num_proposals,
             refined_box_encodings.shape[1],
             self._box_coder.code_size])
        class_predictions_with_background_batch = tf.reshape(
            class_predictions_with_background,
            [-1, self.max_num_proposals, self.num_classes + 1]
        )
        refined_decoded_boxes_batch = self._batch_decode_boxes(
            refined_box_encodings_batch, proposal_boxes)
        class_predictions_with_background_batch = (
            self._second_stage_score_conversion_fn(
                class_predictions_with_background_batch))
        class_predictions_batch = tf.reshape(
            tf.slice(class_predictions_with_background_batch,
                     [0, 0, 1], [-1, -1, -1]),
            [-1, self.max_num_proposals, self.num_classes])
        clip_window = self._compute_clip_window(image_shapes)
        mask_predictions_batch = None
        if mask_predictions is not None:
            mask_height = mask_predictions.shape[2].value
            mask_width = mask_predictions.shape[3].value
            mask_predictions = tf.sigmoid(mask_predictions)
            mask_predictions_batch = tf.reshape(
                mask_predictions, [-1, self.max_num_proposals,
                                   self.num_classes, mask_height, mask_width])

        (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks, _,
         num_detections) = self._second_stage_nms_fn(
            refined_decoded_boxes_batch,
            class_predictions_batch,
            clip_window=clip_window,
            change_coordinate_frame=True,
            num_valid_boxes=num_proposals,
            masks=mask_predictions_batch)
        detections = {
            fields.DetectionResultFields.detection_boxes: nmsed_boxes,
            fields.DetectionResultFields.detection_scores: nmsed_scores,
            fields.DetectionResultFields.detection_classes: nmsed_classes,
            fields.DetectionResultFields.num_detections: tf.to_float(
                num_detections)
        }
        if nmsed_masks is not None:
            detections[
                fields.DetectionResultFields.detection_masks] = nmsed_masks
        return detections

    def _batch_decode_boxes(self, box_encodings, anchor_boxes):
        """Decodes box encodings with respect to the anchor boxes.

        Args:
          box_encodings: a 4-D tensor with shape
            [batch_size, num_anchors, num_classes, self._box_coder.code_size]
            representing box encodings.
          anchor_boxes: [batch_size, num_anchors, self._box_coder.code_size]
            representing decoded bounding boxes. If using a shared box across
            classes the shape will instead be
            [total_num_proposals, 1, self._box_coder.code_size].

        Returns:
          decoded_boxes: a
            [batch_size, num_anchors, num_classes, self._box_coder.code_size]
            float tensor representing bounding box predictions (for each image in
            batch, proposal and class). If using a shared box across classes the
            shape will instead be
            [batch_size, num_anchors, 1, self._box_coder.code_size].
        """
        combined_shape = shape_utils.combined_static_and_dynamic_shape(
            box_encodings)
        num_classes = combined_shape[2]
        tiled_anchor_boxes = tf.tile(
            tf.expand_dims(anchor_boxes, 2), [1, 1, num_classes, 1])
        tiled_anchors_boxlist = box_list.BoxList(
            tf.reshape(tiled_anchor_boxes, [-1, 4]))
        decoded_boxes = self._box_coder.decode(
            tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
            tiled_anchors_boxlist)
        return tf.reshape(decoded_boxes.get(),
                          tf.stack([combined_shape[0], combined_shape[1],
                                    num_classes, 4]))

    def _format_groundtruth_data(self, true_image_shapes):
        """Helper function for preparing groundtruth data for target assignment.

        In order to be consistent with the model.DetectionModel interface,
        groundtruth boxes are specified in normalized coordinates and classes are
        specified as label indices with no assumed background category.  To prepare
        for target assignment, we:
        1) convert boxes to absolute coordinates,
        2) add a background class at class index 0
        3) groundtruth instance masks, if available, are resized to match
           image_shape.

        Args:
          true_image_shapes: int32 tensor of shape [batch, 3] where each row is
            of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be padded
            with zeros.

        Returns:
          groundtruth_boxlists: A list of BoxLists containing (absolute) coordinates
            of the groundtruth boxes.
          groundtruth_classes_with_background_list: A list of 2-D one-hot
            (or k-hot) tensors of shape [num_boxes, num_classes+1] containing the
            class targets with the 0th index assumed to map to the background class.
          groundtruth_masks_list: If present, a list of 3-D tf.float32 tensors of
            shape [num_boxes, image_height, image_width] containing instance masks.
            This is set to None if no masks exist in the provided groundtruth.
        """
        groundtruth_boxlists = [
            box_list_ops.to_absolute_coordinates(
                box_list.BoxList(boxes), true_image_shapes[i, 0],
                true_image_shapes[i, 1])
            for i, boxes in enumerate(
                self.groundtruth_lists(fields.BoxListFields.boxes))
        ]
        groundtruth_classes_with_background_list = [
            tf.to_float(
                tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT'))
            for one_hot_encoding in self.groundtruth_lists(
                fields.BoxListFields.classes)]

        groundtruth_masks_list = self._groundtruth_lists.get(
            fields.BoxListFields.masks)
        # TODO(rathodv): Remove mask resizing once the legacy pipeline is deleted.
        if groundtruth_masks_list is not None and self._resize_masks:
            resized_masks_list = []
            for mask in groundtruth_masks_list:
                _, resized_mask, _ = self._image_resizer_fn(
                    # Reuse the given `image_resizer_fn` to resize groundtruth masks.
                    # `mask` tensor for an image is of the shape [num_masks,
                    # image_height, image_width]. Below we create a dummy image of the
                    # the shape [image_height, image_width, 1] to use with
                    # `image_resizer_fn`.
                    image=tf.zeros(tf.stack([tf.shape(mask)[1],
                                             tf.shape(mask)[2], 1])),
                    masks=mask)
                resized_masks_list.append(resized_mask)

            groundtruth_masks_list = resized_masks_list
        if self.groundtruth_has_field(fields.BoxListFields.weights):
            groundtruth_weights_list = self.groundtruth_lists(
                fields.BoxListFields.weights)
        else:
            # Set weights for all batch elements equally to 1.0
            groundtruth_weights_list = []
            for groundtruth_classes in groundtruth_classes_with_background_list:
                num_gt = tf.shape(groundtruth_classes)[0]
                groundtruth_weights = tf.ones(num_gt)
                groundtruth_weights_list.append(groundtruth_weights)

        return (groundtruth_boxlists, groundtruth_classes_with_background_list,
                groundtruth_masks_list, groundtruth_weights_list)

    def loss(self, prediction_dict, true_image_shapes, scope=None):
        """Compute scalar loss tensors given prediction tensors.

        If number_of_stages=1, only RPN related losses are computed (i.e.,
        `rpn_localization_loss` and `rpn_objectness_loss`).  Otherwise all
        losses are computed.

        Args:
          prediction_dict: a dictionary holding prediction tensors (see the
            documentation for the predict method.  If number_of_stages=1, we
            expect prediction_dict to contain `rpn_box_encodings`,
            `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
            `image_shape`, and `anchors` fields.  Otherwise we expect
            prediction_dict to additionally contain `refined_box_encodings`,
            `class_predictions_with_background`, `num_proposals`, and
            `proposal_boxes` fields.
          true_image_shapes: int32 tensor of shape [batch, 3] where each row is
            of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be padded
            with zeros.
          scope: Optional scope name.

        Returns:
          a dictionary mapping loss keys (`first_stage_localization_loss`,
            `first_stage_objectness_loss`, 'second_stage_localization_loss',
            'second_stage_classification_loss') to scalar tensors representing
            corresponding loss values.
        """

        (groundtruth_boxlists, groundtruth_classes_with_background_list,
         groundtruth_masks_list, groundtruth_weights_list
         ) = self._format_groundtruth_data(true_image_shapes)
        prediction_dict_coarse = prediction_dict['coarse']
        with tf.name_scope(scope, 'CoarseStageLoss', prediction_dict.values()):
            loss_dict = self._loss_box_classifier(
                    prediction_dict_coarse['refined_box_encodings'],
                    prediction_dict_coarse['class_predictions_with_background'],
                    prediction_dict_coarse['proposal_boxes'],
                    prediction_dict_coarse['num_proposals'],
                    groundtruth_boxlists,
                    groundtruth_classes_with_background_list,
                    groundtruth_weights_list,
                    prediction_dict['image_shape'],
                    prediction_dict.get('mask_predictions'),
                    groundtruth_masks_list
                )
        prediction_dict_fine = prediction_dict['fine']
        with tf.name_scope(scope, 'FineStageLoss', prediction_dict.values()):
            loss_dict.update(self._loss_box_classifier(
                    prediction_dict_fine['refined_box_encodings'],
                    prediction_dict_fine['class_predictions_with_background'],
                    prediction_dict_fine['proposal_boxes'],
                    prediction_dict_fine['num_proposals'],
                    groundtruth_boxlists,
                    groundtruth_classes_with_background_list,
                    groundtruth_weights_list,
                    prediction_dict['image_shape'],
                    prediction_dict.get('mask_predictions'),
                    groundtruth_masks_list
                ))

        return loss_dict

    def _loss_box_classifier(self,
                             refined_box_encodings,
                             class_predictions_with_background,
                             proposal_boxes,
                             num_proposals,
                             groundtruth_boxlists,
                             groundtruth_classes_with_background_list,
                             groundtruth_weights_list,
                             image_shape,
                             detector_target_assigner,
                             prediction_masks=None,
                             groundtruth_masks_list=None):
        """Computes scalar box classifier loss tensors.

        Uses self._detector_target_assigner to obtain regression and classification
        targets for the second stage box classifier, optionally performs
        hard mining, and returns losses.  All losses are computed independently
        for each image and then averaged across the batch.
        Please note that for boxes and masks with multiple labels, the box
        regression and mask prediction losses are only computed for one label.

        This function assumes that the proposal boxes in the "padded" regions are
        actually zero (and thus should not be matched to).


        Args:
          refined_box_encodings: a 3-D tensor with shape
            [total_num_proposals, num_classes, box_coder.code_size] representing
            predicted (final) refined box encodings. If using a shared box across
            classes this will instead have shape
            [total_num_proposals, 1, box_coder.code_size].
          class_predictions_with_background: a 2-D tensor with shape
            [total_num_proposals, num_classes + 1] containing class
            predictions (logits) for each of the anchors.  Note that this tensor
            *includes* background class predictions (at class index 0).
          proposal_boxes: [batch_size, self.max_num_proposals, 4] representing
            decoded proposal bounding boxes.
          num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
            representing the number of proposals predicted for each image in
            the batch.
          groundtruth_boxlists: a list of BoxLists containing coordinates of the
            groundtruth boxes.
          groundtruth_classes_with_background_list: a list of 2-D one-hot
            (or k-hot) tensors of shape [num_boxes, num_classes + 1] containing the
            class targets with the 0th index assumed to map to the background class.
          groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
            [num_boxes] containing weights for groundtruth boxes.
          image_shape: a 1-D tensor of shape [4] representing the image shape.
          prediction_masks: an optional 4-D tensor with shape [total_num_proposals,
            num_classes, mask_height, mask_width] containing the instance masks for
            each box.
          groundtruth_masks_list: an optional list of 3-D tensors of shape
            [num_boxes, image_height, image_width] containing the instance masks for
            each of the boxes.

        Returns:
          a dictionary mapping loss keys ('second_stage_localization_loss',
            'second_stage_classification_loss') to scalar tensors representing
            corresponding loss values.

        Raises:
          ValueError: if `predict_instance_masks` in
            second_stage_mask_rcnn_box_predictor is True and
            `groundtruth_masks_list` is not provided.
        """
        with tf.name_scope('BoxClassifierLoss'):
            paddings_indicator = self._padded_batched_proposals_indicator(
                num_proposals, self.max_anchors)
            proposal_boxlists = [
                box_list.BoxList(proposal_boxes_single_image)
                for proposal_boxes_single_image in tf.unstack(proposal_boxes)]
            batch_size = len(proposal_boxlists)

            num_proposals_or_one = tf.to_float(tf.expand_dims(
                tf.maximum(num_proposals, tf.ones_like(num_proposals)), 1))
            normalizer = tf.tile(num_proposals_or_one,
                                 [1, self.max_num_proposals]) * batch_size

            (batch_cls_targets_with_background, batch_cls_weights,
             batch_reg_targets,
             batch_reg_weights, _) = target_assigner.batch_assign_targets(
                target_assigner=detector_target_assigner,
                anchors_batch=proposal_boxlists,
                gt_box_batch=groundtruth_boxlists,
                gt_class_targets_batch=groundtruth_classes_with_background_list,
                unmatched_class_label=tf.constant(
                    [1] + self._num_classes * [0], dtype=tf.float32),
                gt_weights_batch=groundtruth_weights_list)

            class_predictions_with_background = tf.reshape(
                class_predictions_with_background,
                [batch_size, self.max_num_proposals, -1])

            flat_cls_targets_with_background = tf.reshape(
                batch_cls_targets_with_background,
                [batch_size * self.max_num_proposals, -1])
            one_hot_flat_cls_targets_with_background = tf.argmax(
                flat_cls_targets_with_background, axis=1)
            one_hot_flat_cls_targets_with_background = tf.one_hot(
                one_hot_flat_cls_targets_with_background,
                flat_cls_targets_with_background.get_shape()[1])

            # If using a shared box across classes use directly
            if refined_box_encodings.shape[1] == 1:
                reshaped_refined_box_encodings = tf.reshape(
                    refined_box_encodings,
                    [batch_size, self.max_anchors,
                     self._box_coder.code_size])
            # For anchors with multiple labels, picks refined_location_encodings
            # for just one class to avoid over-counting for regression loss and
            # (optionally) mask loss.
            else:
                reshaped_refined_box_encodings = (
                    self._get_refined_encodings_for_postitive_class(
                        refined_box_encodings,
                        one_hot_flat_cls_targets_with_background, batch_size))

            losses_mask = None
            if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
                losses_mask = tf.stack(self.groundtruth_lists(
                    fields.InputDataFields.is_annotated))
            second_stage_loc_losses = self._second_stage_localization_loss(
                reshaped_refined_box_encodings,
                batch_reg_targets,
                weights=batch_reg_weights,
                losses_mask=losses_mask) / normalizer
            second_stage_cls_losses = ops.reduce_sum_trailing_dimensions(
                self._second_stage_classification_loss(
                    class_predictions_with_background,
                    batch_cls_targets_with_background,
                    weights=batch_cls_weights,
                    losses_mask=losses_mask),
                ndims=2) / normalizer

            second_stage_loc_loss = tf.reduce_sum(
                second_stage_loc_losses * tf.to_float(paddings_indicator))
            second_stage_cls_loss = tf.reduce_sum(
                second_stage_cls_losses * tf.to_float(paddings_indicator))

            if self._hard_example_miner:
                (second_stage_loc_loss, second_stage_cls_loss
                 ) = self._unpad_proposals_and_apply_hard_mining(
                    proposal_boxlists, second_stage_loc_losses,
                    second_stage_cls_losses, num_proposals)
            localization_loss = tf.multiply(self._second_stage_loc_loss_weight,
                                            second_stage_loc_loss,
                                            name='localization_loss')

            classification_loss = tf.multiply(
                self._second_stage_cls_loss_weight,
                second_stage_cls_loss,
                name='classification_loss')

            loss_dict = {localization_loss.op.name: localization_loss,
                         classification_loss.op.name: classification_loss}
            second_stage_mask_loss = None
            if prediction_masks is not None:
                if groundtruth_masks_list is None:
                    raise ValueError('Groundtruth instance masks not provided. '
                                     'Please configure input reader.')

                unmatched_mask_label = tf.zeros(image_shape[1:3],
                                                dtype=tf.float32)
                (batch_mask_targets, _, _, batch_mask_target_weights,
                 _) = target_assigner.batch_assign_targets(
                    target_assigner=self._detector_target_assigner,
                    anchors_batch=proposal_boxlists,
                    gt_box_batch=groundtruth_boxlists,
                    gt_class_targets_batch=groundtruth_masks_list,
                    unmatched_class_label=unmatched_mask_label,
                    gt_weights_batch=groundtruth_weights_list)

                # Pad the prediction_masks with to add zeros for background class to be
                # consistent with class predictions.
                if prediction_masks.get_shape().as_list()[1] == 1:
                    # Class agnostic masks or masks for one-class prediction. Logic for
                    # both cases is the same since background predictions are ignored
                    # through the batch_mask_target_weights.
                    prediction_masks_masked_by_class_targets = prediction_masks
                else:
                    prediction_masks_with_background = tf.pad(
                        prediction_masks, [[0, 0], [1, 0], [0, 0], [0, 0]])
                    prediction_masks_masked_by_class_targets = tf.boolean_mask(
                        prediction_masks_with_background,
                        tf.greater(one_hot_flat_cls_targets_with_background, 0))

                mask_height = prediction_masks.shape[2].value
                mask_width = prediction_masks.shape[3].value
                reshaped_prediction_masks = tf.reshape(
                    prediction_masks_masked_by_class_targets,
                    [batch_size, -1, mask_height * mask_width])

                batch_mask_targets_shape = tf.shape(batch_mask_targets)
                flat_gt_masks = tf.reshape(batch_mask_targets,
                                           [-1, batch_mask_targets_shape[2],
                                            batch_mask_targets_shape[3]])

                # Use normalized proposals to crop mask targets from image masks.
                flat_normalized_proposals = box_list_ops.to_normalized_coordinates(
                    box_list.BoxList(tf.reshape(proposal_boxes, [-1, 4])),
                    image_shape[1], image_shape[2]).get()

                flat_cropped_gt_mask = self._crop_and_resize_fn(
                    tf.expand_dims(flat_gt_masks, -1),
                    tf.expand_dims(flat_normalized_proposals, axis=1),
                    [mask_height, mask_width])

                batch_cropped_gt_mask = tf.reshape(
                    flat_cropped_gt_mask,
                    [batch_size, -1, mask_height * mask_width])

                second_stage_mask_losses = ops.reduce_sum_trailing_dimensions(
                    self._second_stage_mask_loss(
                        reshaped_prediction_masks,
                        batch_cropped_gt_mask,
                        weights=batch_mask_target_weights,
                        losses_mask=losses_mask),
                    ndims=2) / (
                                                   mask_height * mask_width * tf.maximum(
                                               tf.reduce_sum(
                                                   batch_mask_target_weights,
                                                   axis=1, keep_dims=True
                                               ), tf.ones((batch_size, 1))))
                second_stage_mask_loss = tf.reduce_sum(
                    tf.where(paddings_indicator, second_stage_mask_losses,
                             tf.zeros_like(second_stage_mask_losses)))

            if second_stage_mask_loss is not None:
                mask_loss = tf.multiply(self._second_stage_mask_loss_weight,
                                        second_stage_mask_loss,
                                        name='mask_loss')
                loss_dict[mask_loss.op.name] = mask_loss
        return loss_dict

    def _get_refined_encodings_for_postitive_class(
            self, refined_box_encodings, flat_cls_targets_with_background,
            batch_size):
        # We only predict refined location encodings for the non background
        # classes, but we now pad it to make it compatible with the class
        # predictions
        refined_box_encodings_with_background = tf.pad(refined_box_encodings,
                                                       [[0, 0], [1, 0], [0, 0]])
        refined_box_encodings_masked_by_class_targets = (
            box_list_ops.boolean_mask(
                box_list.BoxList(
                    tf.reshape(refined_box_encodings_with_background,
                               [-1, self._box_coder.code_size])),
                tf.reshape(tf.greater(flat_cls_targets_with_background, 0),
                           [-1]),
                use_static_shapes=self._use_static_shapes,
                indicator_sum=batch_size * self.max_num_proposals
                if self._use_static_shapes else None).get())
        return tf.reshape(
            refined_box_encodings_masked_by_class_targets, [
                batch_size, self.max_num_proposals,
                self._box_coder.code_size
            ])

    def _padded_batched_proposals_indicator(self,
                                            num_proposals,
                                            max_num_proposals):
        """Creates indicator matrix of non-pad elements of padded batch proposals.

        Args:
          num_proposals: Tensor of type tf.int32 with shape [batch_size].
          max_num_proposals: Maximum number of proposals per image (integer).

        Returns:
          A Tensor of type tf.bool with shape [batch_size, max_num_proposals].
        """
        batch_size = tf.size(num_proposals)
        tiled_num_proposals = tf.tile(
            tf.expand_dims(num_proposals, 1), [1, max_num_proposals])
        tiled_proposal_index = tf.tile(
            tf.expand_dims(tf.range(max_num_proposals), 0), [batch_size, 1])
        return tf.greater(tiled_num_proposals, tiled_proposal_index)

    def _unpad_proposals_and_apply_hard_mining(self,
                                               proposal_boxlists,
                                               second_stage_loc_losses,
                                               second_stage_cls_losses,
                                               num_proposals):
        """Unpads proposals and applies hard mining.

        Args:
          proposal_boxlists: A list of `batch_size` BoxLists each representing
            `self.max_num_proposals` representing decoded proposal bounding boxes
            for each image.
          second_stage_loc_losses: A Tensor of type `float32`. A tensor of shape
            `[batch_size, self.max_num_proposals]` representing per-anchor
            second stage localization loss values.
          second_stage_cls_losses: A Tensor of type `float32`. A tensor of shape
            `[batch_size, self.max_num_proposals]` representing per-anchor
            second stage classification loss values.
          num_proposals: A Tensor of type `int32`. A 1-D tensor of shape [batch]
            representing the number of proposals predicted for each image in
            the batch.

        Returns:
          second_stage_loc_loss: A scalar float32 tensor representing the second
            stage localization loss.
          second_stage_cls_loss: A scalar float32 tensor representing the second
            stage classification loss.
        """
        for (proposal_boxlist, single_image_loc_loss, single_image_cls_loss,
             single_image_num_proposals) in zip(
            proposal_boxlists,
            tf.unstack(second_stage_loc_losses),
            tf.unstack(second_stage_cls_losses),
            tf.unstack(num_proposals)):
            proposal_boxlist = box_list.BoxList(
                tf.slice(proposal_boxlist.get(),
                         [0, 0], [single_image_num_proposals, -1]))
            single_image_loc_loss = tf.slice(single_image_loc_loss,
                                             [0], [single_image_num_proposals])
            single_image_cls_loss = tf.slice(single_image_cls_loss,
                                             [0], [single_image_num_proposals])
            return self._hard_example_miner(
                location_losses=tf.expand_dims(single_image_loc_loss, 0),
                cls_losses=tf.expand_dims(single_image_cls_loss, 0),
                decoded_boxlist_list=[proposal_boxlist])

    def restore_map(self,
                    fine_tune_checkpoint_type='detection',
                    load_all_detection_checkpoint_vars=False):
        """Returns a map of variables to load from a foreign checkpoint.

        See parent class for details.

        Args:
          fine_tune_checkpoint_type: whether to restore from a full detection
            checkpoint (with compatible variable names) or to restore from a
            classification checkpoint for initialization prior to training.
            Valid values: `detection`, `classification`. Default 'detection'.
           load_all_detection_checkpoint_vars: whether to load all variables (when
             `fine_tune_checkpoint_type` is `detection`). If False, only variables
             within the feature extractor scopes are included. Default False.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        Raises:
          ValueError: if fine_tune_checkpoint_type is neither `classification`
            nor `detection`.
        """
        if fine_tune_checkpoint_type not in ['detection', 'classification']:
            raise ValueError(
                'Not supported fine_tune_checkpoint_type: {}'.format(
                    fine_tune_checkpoint_type))
        if fine_tune_checkpoint_type == 'classification':
            return self._feature_extractor.restore_from_classification_checkpoint_fn(
                self.first_stage_feature_extractor_scope,
                self.second_stage_feature_extractor_scope)

        variables_to_restore = tf.global_variables()
        variables_to_restore.append(slim.get_or_create_global_step())
        # Only load feature extractor variables to be consistent with loading from
        # a classification checkpoint.
        include_patterns = None
        if not load_all_detection_checkpoint_vars:
            include_patterns = [
                self.first_stage_feature_extractor_scope,
                self.second_stage_feature_extractor_scope
            ]
        feature_extractor_variables = tf.contrib.framework.filter_variables(
            variables_to_restore, include_patterns=include_patterns)
        return {var.op.name: var for var in feature_extractor_variables}


    def postprocess(self, prediction_dict, true_image_shapes):
        """Convert prediction tensors to final detections.

        This function converts raw predictions tensors to final detection results.
        See base class for output format conventions.  Note also that by default,
        scores are to be interpreted as logits, but if a score_converter is used,
        then scores are remapped (and may thus have a different interpretation).

        If number_of_stages=1, the returned results represent proposals from the
        first stage RPN and are padded to have self.max_num_proposals for each
        image; otherwise, the results can be interpreted as multiclass detections
        from the full two-stage model and are padded to self._max_detections.

        Args:
          prediction_dict: a dictionary holding prediction tensors (see the
            documentation for the predict method.  If number_of_stages=1, we
            expect prediction_dict to contain `rpn_box_encodings`,
            `rpn_objectness_predictions_with_background`, `rpn_features_to_crop`,
            and `anchors` fields.  Otherwise we expect prediction_dict to
            additionally contain `refined_box_encodings`,
            `class_predictions_with_background`, `num_proposals`,
            `proposal_boxes` and, optionally, `mask_predictions` fields.
          true_image_shapes: int32 tensor of shape [batch, 3] where each row is
            of the form [height, width, channels] indicating the shapes
            of true images in the resized images, as resized images can be padded
            with zeros.

        Returns:
          detections: a dictionary containing the following fields
            detection_boxes: [batch, max_detection, 4]
            detection_scores: [batch, max_detections]
            detection_classes: [batch, max_detections]
              (this entry is only created if rpn_mode=False)
            num_detections: [batch]

        Raises:
          ValueError: If `predict` is called before `preprocess`.
        """

        # with tf.name_scope('FirstStagePostprocessor'):
        #   if self._number_of_stages == 1:
        #     proposal_boxes, proposal_scores, num_proposals = self._postprocess_rpn(
        #         prediction_dict['rpn_box_encodings'],
        #         prediction_dict['rpn_objectness_predictions_with_background'],
        #         prediction_dict['anchors'],
        #         true_image_shapes,
        #         true_image_shapes)
        #     return {
        #         fields.DetectionResultFields.detection_boxes: proposal_boxes,
        #         fields.DetectionResultFields.detection_scores: proposal_scores,
        #         fields.DetectionResultFields.num_detections:
        #             tf.to_float(num_proposals),
        #     }

        # TODO(jrru): Remove mask_predictions from _post_process_box_classifier.
        with tf.name_scope('FineStagePostprocessor'):
          mask_predictions = prediction_dict.get(box_predictor.MASK_PREDICTIONS)
          detections_dict = self._postprocess_box_classifier(
            prediction_dict['refined_box_encodings'],
            prediction_dict['class_predictions_with_background'],
            prediction_dict['proposal_boxes'],
            prediction_dict['num_proposals'],
            true_image_shapes,
            mask_predictions=mask_predictions)
        return detections_dict

        # if self._number_of_stages == 3:
        #   # Post processing is already performed in 3rd stage. We need to transfer
        #   # postprocessed tensors from `prediction_dict` to `detections_dict`.
        #   detections_dict = {}
        #   for key in prediction_dict:
        #     if key == fields.DetectionResultFields.detection_masks:
        #       detections_dict[key] = tf.sigmoid(prediction_dict[key])
        #     elif 'detection' in key:
        #       detections_dict[key] = prediction_dict[key]
        #   return detections_dict