from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES_conv = [
    'conv_421',
    'conv_622',
    'conv_823',
]

PRIMITIVES_upconv = [

    're_conv_421',
    're_conv_622',
    're_conv_823',
]


