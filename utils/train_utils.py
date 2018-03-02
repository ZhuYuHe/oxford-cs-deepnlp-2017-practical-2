import tensorflow as tf 

def get_config_proto(log_device_placement = True, allow_soft_placement = True,
                     num_intra_threads = 0, num_inter_threads = 0):
    #GPU options
    config_proto = tf.ConfigProto(
        log_device_placement = log_device_placement,
        allow_soft_placement = allow_soft_placement
    )
    config_proto.gpu_options.allow_growth = True

    #CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto