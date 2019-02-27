import tensorflow as tf
from object_detection.protos import spatial_attention_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.builders import hyperparams_builder

slim = tf.contrib.slim

def build(spatial_attention_config, is_training):
    depthwise_separable_layer_scope_fn  = hyperparams_builder.build(
        spatial_attention_config.depthwise_conv, is_training)
    deformable_conv_layer_scope_fn  = hyperparams_builder.build(
        spatial_attention_config.deformable_conv)
    semantic_attention_layer_scope_fn  = hyperparams_builder.build(
        spatial_attention_config.semantic_attention)
    attention_combiner_scope_fn  = hyperparams_builder.build(
        spatial_attention_config.attention_combiner)
    attention_reducer_scope_fn  = hyperparams_builder.build(
        spatial_attention_config.attention_reducer)
    return depthwise_separable_layer_scope_fn,
    deformable_conv_layer_scope_fn,
    semantic_attention_layer_scope_fn,
    attention_combiner_scope_fn,
    attention_reducer_scope_fn

    
    
    
