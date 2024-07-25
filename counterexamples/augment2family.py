from counterexamples.basic_utils import sample, get_complementary_policy, get_all_policy
import numpy as np

number_of_layers = 3
layer_types = ["ori", "com", "all"]
rel_sizes = [0.5, 1.0, 1.5]


def augment2family(policy, generator):
    PCl, PCr = np.zeros((0, 3)), np.zeros((0, 3))
    for i in range(number_of_layers):
        layer_type = layer_types[i]
        if layer_type == "ori":
            the_policy = policy
        elif layer_type == "com":
            the_policy = get_complementary_policy(policy)
        elif layer_type == "all":
            the_policy = get_all_policy(policy)
        
        rel_size = rel_sizes[i]
        the_reg = generator(rel_size)
            
        the_PCl, the_PCr = sample(the_reg, the_policy)
        PCl, PCr = np.concatenate((PCl, the_PCl)), np.concatenate((PCr, the_PCr))
    return PCl, PCr