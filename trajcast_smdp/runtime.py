"""Optional execution optimizations; never alter irreps, weights or dtype."""
import torch
from e3nn import o3


def compile_tensor_products(model):
    """Fuse generated e3nn tensor algebra, preserving checkpoint key names.

    Construct with e3nn jit_mode='eager' first. Only tensor-product programs
    are compiled; graph topology and graph sizes are still handled upstream.
    Dynamic shapes permit varying periodic edge counts without specialization
    to a fixed Quartz atom count. Failure is explicit, not silent fallback.
    """
    names=[]
    for name,module in model.named_modules():
        if isinstance(module,o3.TensorProduct):
            program=module._compiled_main_left_right
            if isinstance(program,torch.jit.ScriptModule):
                raise ValueError('Tensor-product fusion requires eager e3nn code generation at construction')
            program.forward=torch.compile(program.forward,dynamic=True,fullgraph=True)
            names.append(name)
    if not names: raise ValueError('No tensor products found')
    return names
