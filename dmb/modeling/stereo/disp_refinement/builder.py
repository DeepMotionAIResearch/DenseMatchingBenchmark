from .StereoNet import StereoNetRefinement
from .DeepPruner import DeepPrunerRefinement
from .AnyNet import AnyNetRefinement

REFINEMENTS = {
    "StereoNet": StereoNetRefinement,
    "DeepPruner": DeepPrunerRefinement,
    "AnyNet": AnyNetRefinement,
}


def build_disp_refinement(cfg):
    refine_type = cfg.model.disp_refinement.type
    assert refine_type in REFINEMENTS, "disp refinement type not found, excepted: {}," \
                                     "but got {}".format(REFINEMENTS.keys(), refine_type)

    default_args = cfg.model.disp_refinement.copy()
    default_args.pop('type')
    default_args.update(batch_norm=cfg.model.batch_norm)

    refinement = REFINEMENTS[refine_type](**default_args)

    return refinement
