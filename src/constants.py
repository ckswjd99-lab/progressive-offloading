SERVER_IP = '127.0.0.1'
SERVER_PORT_ES    = 61234
SERVER_PORT_SE    = SERVER_PORT_ES + 1
SERVER_PORT_META  = SERVER_PORT_SE + 1

SELF_IP = '127.0.0.1'

NUM_REPEATS = 10

HEADER_CLOCK = 999
HEADER_FINISH = 1000
HEADER_IMAGE = 1001
HEADER_RESULT = 1002

DTYPE_FP64, DTYPE_FP32, DTYPE_FP16, DTYPE_INT32, DTYPE_INT16, DTYPE_INT8, DTYPE_UINT8 = range(7)

EDGE_MODEL_NAME = 'efficientvit_b0.r224_in1k'
# EDGE_MODEL_NAME = 'repghostnet_080.in1k'
# EDGE_MODEL_NAME = 'tf_mobilenetv3_large_075.in1k'
# EDGE_MODEL_NAME = 'mobilenetv3_small_050.lamb_in1k'
# EDGE_MODEL_NAME = 'efficientvit_m1.r224_in1k'
# EDGE_MODEL_NAME = 'vgg11_bn.tv_in1k'


# SERVER_MODEL_NAME = 'caformer_b36.sail_in22k_ft_in1k'
# SERVER_MODEL_NAME = 'caformer_m36.sail_in22k_ft_in1k'
SERVER_MODEL_NAME = 'caformer_s36.sail_in22k_ft_in1k'

# SERVER_MODEL_NAME = 'vit_huge_patch14_clip_224.laion2b_ft_in12k_in1k'
# SERVER_MODEL_NAME = 'vit_large_patch14_clip_224.openai_ft_in12k_in1k'
# SERVER_MODEL_NAME = 'vit_base_patch8_224.augreg2_in21k_ft_in1k'

# SERVER_MODEL_NAME = 'convformer_b36.sail_in22k_ft_in1k'
# SERVER_MODEL_NAME = 'convformer_m36.sail_in22k_ft_in1k'
# SERVER_MODEL_NAME = 'convformer_s36.sail_in22k_ft_in1k'

# SERVER_MODEL_NAME = 'deit3_huge_patch14_224.fb_in22k_ft_in1k'
# SERVER_MODEL_NAME = 'deit3_large_patch16_224.fb_in22k_ft_in1k'
# SERVER_MODEL_NAME = 'deit3_medium_patch16_224.fb_in22k_ft_in1k'
# SERVER_MODEL_NAME = 'deit3_small_patch16_224.fb_in22k_ft_in1k'