import tensorflow as tf
from object_detection.protos import spatial_attention_pb2
from object_detection.protos import hyperparams_pb2
from object_detection.builders import hyperparams_builder

slim = tf.contrib.slim

def build(spatial_attention_config, is_training):
    semantic_attention_layer_scope_fn = []
    for semantic_attention_config in spatial_attention_config.semantic_attention:
        semantic_attention_layer_scope_fn.append(
            hyperparams_builder.build(semantic_attention_config, is_training)
            )

    attention_combiner_scope_fn  = hyperparams_builder.build(spatial_attention_config.attention_combiner, is_training)
    attention_reducer_scope_fn  = hyperparams_builder.build(spatial_attention_config.attention_reducer, is_training)
    return semantic_attention_layer_scope_fn, attention_combiner_scope_fn, attention_reducer_scope_fn

    
    
    
