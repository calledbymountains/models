import abc
import tensorflow as tf
from object_detection.core import model
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.utils import shape_utils

slim = tf.contrib.slim


class IJCAIFeatureExtractor(object):
    def __init__(self, name, is_training,
                 output_stride, batch_norm_trainable=False,
                 reuse_weights=None, weight_decay=0.0):
        """Constructor.

            Args:
              is_training: A boolean indicating whether the training version of the
                computation graph should be constructed.
              output_stride: Output stride of extracted feature map.
              batch_norm_trainable: Whether to update batch norm parameters during
                training or not. When training with a relative large batch size
                (e.g. 8), it could be desirable to enable batch norm update.
              reuse_weights: Whether to reuse variables. Default is None.
              weight_decay: float weight decay for feature extractor (default: 0.0).
            """
        self._name = name
        self._is_training = is_training
        self._train_batch_norm = batch_norm_trainable
        self._output_stride = output_stride
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay

    @property
    def is_training(self):
        return self._is_training

    @property
    def output_stride(self):
        return self._output_stride

    @abc.abstractmethod
    def preprocess(self, resized_inputs):
        """Feature-extractor specific preprocessing (minus image resizing)."""
        pass

    def extract_anchor_features(self, preprocessed_inputs, scope):
        """Extracts first stage features.

        This function is responsible for extracting feature maps from preprocessed
        images.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float tensor
            representing a batch of images.
          scope: A scope name.

        Returns:
          feature_map: A tensor with shape [batch, height, width, depth]
          activations: A dictionary mapping activation tensor names to tensors.
        """
        with tf.variable_scope(scope, values=[preprocessed_inputs]):
            return self._extract_anchor_features(preprocessed_inputs, scope)

    @abc.abstractmethod
    def _extract_anchor_features(self, preprocessed_inputs, scope):
        """Extracts first stage features, to be overridden."""
        pass

    def extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features.

        Args:
          proposal_feature_maps: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
            representing the feature map cropped to each proposal.
          scope: A scope name.

        Returns:
          proposal_classifier_features: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, height, width, depth]
            representing box classifier features for each proposal.
        """
        with tf.variable_scope(
                scope, values=[proposal_feature_maps], reuse=tf.AUTO_REUSE):
            return self._extract_box_classifier_features(proposal_feature_maps, scope)

    @abc.abstractmethod
    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features, to be overridden."""
        pass

    def restore_from_classification_checkpoint_fn(
            self,
            first_stage_feature_extractor_scope,
            second_stage_feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.

        Args:
          first_stage_feature_extractor_scope: A scope name for the first stage
            feature extractor.
          second_stage_feature_extractor_scope: A scope name for the second stage
            feature extractor.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            for scope_name in [first_stage_feature_extractor_scope,
                               second_stage_feature_extractor_scope]:
                if variable.op.name.startswith(scope_name):
                    var_name = variable.op.name.replace(scope_name + '/', '')
                    variables_to_restore[var_name] = variable
        return variables_to_restore


class IJCAIDetectionModel(model.DetectionModel):
    """IJCAI Detector Meta-architecture definition."""

    def __init__(self, is_training,
                 num_classes,
                 image_resizer_fn,
                 feature_extractor,
                 depthwise_dict,
                 anchor_generator,
                 coarse_target_assigner,
                 coarse_box_predictor_arg_scope_fn,
                 coarse_box_predictor_kernel_size,
                 coarse_box_predictor_depth,
                 fine_target_assigner,
                 fine_box_predictor_arg_scope_fn,
                 fine_box_predictor_kernel_size,
                 fine_box_predictor_depth,
                 anchor_minibatch_size,
                 hard_example_miner=None,
                 clip_anchors_to_image=False,
                 use_static_shapes=False,
                 parallel_iterations=16
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
              feature_extractor: A FasterRCNNFeatureExtractor object.
              coarse_anchor_generator: An anchor_generator.AnchorGenerator object
                (note that currently we only support
                grid_anchor_generator.GridAnchorGenerator objects)
              coarse_target_assigner: Target assigner to use for first stage of
                Faster R-CNN (RPN).
              coarse_atrous_rate: A single integer indicating the atrous rate for
                the single convolution op which is applied to the `rpn_features_to_crop`
                tensor to obtain a tensor to be used for box prediction. Some feature
                extractors optionally allow for producing feature maps computed at
                denser resolutions.  The atrous rate is used to compensate for the
                denser feature maps by using an effectively larger receptive field.
                (This should typically be set to 1).
              coarse_box_predictor_arg_scope_fn: A function to construct tf-slim
                arg_scope for conv2d, separable_conv2d and fully_connected ops for the
                RPN box predictor.
              coarse_box_predictor_kernel_size: Kernel size to use for the
                convolution op just prior to RPN box predictions.
              coarse_box_predictor_depth: Output depth for the convolution op
                just prior to RPN box predictions.
              first_stage_minibatch_size: The "batch size" to use for computing the
                objectness and location loss of the region proposal network. This
                "batch size" refers to the number of anchors selected as contributing
                to the loss function for any given image within the image batch and is
                only called "batch_size" due to terminology from the Faster R-CNN paper.
              first_stage_sampler: Sampler to use for first stage loss (RPN loss).
              first_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
                callable that takes `boxes`, `scores` and optional `clip_window`(with
                all other inputs already set) and returns a dictionary containing
                tensors with keys: `detection_boxes`, `detection_scores`,
                `detection_classes`, `num_detections`. This is used to perform non max
                suppression  on the boxes predicted by the Region Proposal Network
                (RPN).
                See `post_processing.batch_multiclass_non_max_suppression` for the type
                and shape of these tensors.
              first_stage_max_proposals: Maximum number of boxes to retain after
                performing Non-Max Suppression (NMS) on the boxes predicted by the
                Region Proposal Network (RPN).
              first_stage_localization_loss_weight: A float
              first_stage_objectness_loss_weight: A float
              crop_and_resize_fn: A differentiable resampler to use for cropping RPN
                proposal features.
              initial_crop_size: A single integer indicating the output size
                (width and height are set to be the same) of the initial bilinear
                interpolation based cropping during ROI pooling.
              maxpool_kernel_size: A single integer indicating the kernel size of the
                max pool op on the cropped feature map during ROI pooling.
              maxpool_stride: A single integer indicating the stride of the max pool
                op on the cropped feature map during ROI pooling.
              second_stage_target_assigner: Target assigner to use for second stage of
                Faster R-CNN. If the model is configured with multiple prediction heads,
                this target assigner is used to generate targets for all heads (with the
                correct `unmatched_class_label`).
              second_stage_mask_rcnn_box_predictor: Mask R-CNN box predictor to use for
                the second stage.
              second_stage_batch_size: The batch size used for computing the
                classification and refined location loss of the box classifier.  This
                "batch size" refers to the number of proposals selected as contributing
                to the loss function for any given image within the image batch and is
                only called "batch_size" due to terminology from the Faster R-CNN paper.
              second_stage_sampler:  Sampler to use for second stage loss (box
                classifier loss).
              second_stage_non_max_suppression_fn: batch_multiclass_non_max_suppression
                callable that takes `boxes`, `scores`, optional `clip_window` and
                optional (kwarg) `mask` inputs (with all other inputs already set)
                and returns a dictionary containing tensors with keys:
                `detection_boxes`, `detection_scores`, `detection_classes`,
                `num_detections`, and (optionally) `detection_masks`. See
                `post_processing.batch_multiclass_non_max_suppression` for the type and
                shape of these tensors.
              second_stage_score_conversion_fn: Callable elementwise nonlinearity
                (that takes tensors as inputs and returns tensors).  This is usually
                used to convert logits to probabilities.
              second_stage_localization_loss_weight: A float indicating the scale factor
                for second stage localization loss.
              second_stage_classification_loss_weight: A float indicating the scale
                factor for second stage classification loss.
              second_stage_classification_loss: Classification loss used by the second
                stage classifier. Either losses.WeightedSigmoidClassificationLoss or
                losses.WeightedSoftmaxClassificationLoss.
              second_stage_mask_prediction_loss_weight: A float indicating the scale
                factor for second stage mask prediction loss. This is applicable only if
                second stage box predictor is configured to predict masks.
              hard_example_miner:  A losses.HardExampleMiner object (can be None).
              parallel_iterations: (Optional) The number of iterations allowed to run
                in parallel for calls to tf.map_fn.
              add_summaries: boolean (default: True) controlling whether summary ops
                should be added to tensorflow graph.
              clip_anchors_to_image: Normally, anchors generated for a given image size
                are pruned during training if they lie outside the image window. This
                option clips the anchors to be within the image instead of pruning.
              use_static_shapes: If True, uses implementation of ops with static shape
                guarantees.
              resize_masks: Indicates whether the masks presend in the groundtruth
                should be resized in the model with `image_resizer_fn`

            Raises:
              ValueError: If `second_stage_batch_size` > `first_stage_max_proposals` at
                training time.
              ValueError: If first_stage_anchor_generator is not of type
                grid_anchor_generator.GridAnchorGenerator.
            """
        super(IJCAIDetectionModel, self).__init__(num_classes=num_classes)
        if not isinstance(anchor_generator,
                          grid_anchor_generator.GridAnchorGenerator):
            raise ValueError('first_stage_anchor_generator must be of type '
                             'grid_anchor_generator.GridAnchorGenerator.')
        self._image_resizer_fn = image_resizer_fn
        self._feature_extractor = feature_extractor
        self._depthwise_dict = depthwise_dict
        self._anchor_generator = anchor_generator
        self._coarse_target_assigner = coarse_target_assigner
        self._coarse_box_predictor_arg_scope_fn = coarse_box_predictor_arg_scope_fn
        self._coarse_box_predictor_kernel_size = coarse_box_predictor_kernel_size
        self._coarse_box_predictor_depth = coarse_box_predictor_depth
        self._fine_target_assigner = fine_target_assigner
        self._fine_box_predictor_arg_scope_fn = fine_box_predictor_arg_scope_fn
        self._fine_box_predictor_kernel_size = fine_box_predictor_kernel_size
        self._fine_box_predictor_depth = fine_box_predictor_depth
        self._anchor_minibatch_size = anchor_minibatch_size
        self._hard_example_miner = hard_example_miner
        self._clip_anchors_to_image = clip_anchors_to_image
        self._use_static_shapes = use_static_shapes
        self._parallel_iterations = parallel_iterations

    def preprocess(self, inputs):
        """Feature-extractor specific preprocessing.

        See base class.

        For IJCAI Detector, we perform image resizing in the base class --- each
        class subclassing IJCAIDetectorMetaArch is responsible for any additional
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

    @staticmethod
    def _compute_clip_window(image_shapes):
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

    def build_depthwise_separable_layer(self, input_feature_map):
        depthwise_arg_fn = self._depthwise_dict['arg_fn']
        if isinstance(depthwise_arg_fn, dict):
            return input_feature_map

        output_filters = self._depthwise_dict['numfilters']
        depth_multiplier = self._depthwise_dict['depth_multiplier']
        with slim.arg_scope(depthwise_arg_fn):
            output = slim.separable_convolution2d(input_feature_map,
                                                  output_filters,
                                                  3,
                                                  depth_multiplier)
        return output
