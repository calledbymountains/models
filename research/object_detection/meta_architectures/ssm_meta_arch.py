from abc import abstractmethod
import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import visualization_utils

slim = tf.contrib.slim


class SSMFeatureExtractor(object):
    """SSM slim feature extractor"""

    def __init__(self,
                 is_training,
                 output_stride,
                 extraction_points):
        """
        Constructor
        :param is_training: Whether network is in training mode.
        :param output_stride: Output stride of the network.
        :param extraction_points: List of slim endpoints from where feature
        outputs are to be taken
        """
        self._is_training = is_training
        self._output_stride = output_stride
        self._extraction_points = extraction_points

    @abstractmethod
    def preprocess(self, resized_inputs):
        """Preprocesses images for feature extraction (minus image resizing).

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    """
        pass

    @abstractmethod
    def extract_features(self, preprocessed_inputs):
        raise NotImplementedError


class SSMDetectionModel(model.DetectionModel):
    def __init__(self, is_training,
                 ssm_enhancer,
                 dearranger,
                 ssm_rnn,
                 arranger,
                 spatial_attention_coder,
                 anchor_generator,
                 box_predictor,
                 box_coder,
                 feature_extractor,
                 encode_background_as_zeros,
                 image_resizer_fn,
                 non_max_suppression_fn,
                 score_conversion_fn,
                 classification_loss,
                 localization_loss,
                 classification_loss_weight,
                 localization_loss_weight,
                 normalize_loss_by_num_matches,
                 hard_example_miner,
                 target_assigner_instance,
                 add_summaries=True,
                 normalize_loc_loss_by_codesize=False,
                 freeze_batchnorm=False,
                 inplace_batchnorm_update=False,
                 add_background_class=True,
                 random_example_sampler=None,
                 expected_classification_loss_under_sampling=None):
        self._is_training = is_training
        self._ssm_enhancer = ssm_enhancer
        self._dearranger = dearranger
        self._ssm_rnn = ssm_rnn
        self._arranger = arranger
        self._spatial_attention_coder = spatial_attention_coder
        self._freeze_batchnorm = freeze_batchnorm
        self._inplace_batchnorm_update = inplace_batchnorm_update

        self._anchor_generator = anchor_generator
        self._box_predictor = box_predictor

        self._box_coder = box_coder
        self._feature_extractor = feature_extractor
        self._add_background_class = add_background_class

        # Needed for fine-tuning from classification checkpoints whose
        # variables do not have the feature extractor scope.
        if self._feature_extractor.is_keras_model:
            # Keras feature extractors will have a name they implicitly use to scope.
            # So, all contained variables are prefixed by this name.
            # To load from classification checkpoints, need to filter out this name.
            self._extract_features_scope = feature_extractor.name
        else:
            # Slim feature extractors get an explicit naming scope
            self._extract_features_scope = 'FeatureExtractor'

        # TODO(jonathanhuang): handle agnostic mode
        # weights
        self._unmatched_class_label = tf.constant([1] + self.num_classes * [0],
                                                  tf.float32)
        if encode_background_as_zeros:
            self._unmatched_class_label = tf.constant(
                (self.num_classes + 1) * [0],
                tf.float32)

        self._target_assigner = target_assigner_instance

        self._classification_loss = classification_loss
        self._localization_loss = localization_loss
        self._classification_loss_weight = classification_loss_weight
        self._localization_loss_weight = localization_loss_weight
        self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
        self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize
        self._hard_example_miner = hard_example_miner
        self._random_example_sampler = random_example_sampler
        self._parallel_iterations = 16

        self._image_resizer_fn = image_resizer_fn
        self._non_max_suppression_fn = non_max_suppression_fn
        self._score_conversion_fn = score_conversion_fn

        self._anchors = None
        self._add_summaries = add_summaries
        self._batched_prediction_tensor_names = []
        self._expected_classification_loss_under_sampling = (
            expected_classification_loss_under_sampling)

    @property
    def anchors(self):
        if not self._anchors:
            raise RuntimeError('anchors have not been constructed yet!')
        if not isinstance(self._anchors, box_list.BoxList):
            raise RuntimeError(
                'anchors should be a BoxList object, but is not.')
        return self._anchors



