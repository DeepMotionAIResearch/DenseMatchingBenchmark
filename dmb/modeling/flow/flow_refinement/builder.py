from .hm_net import HMRefinement

REFINEMENTS = {
    'HMNet': HMRefinement,
}


def build_flow_refinement(cfg):
    refine_type = cfg.model.flow_refinement.type
    assert refine_type in REFINEMENTS, "flow refinement type not found, excepted: {}," \
                                     "but got {}".format(REFINEMENTS.keys(), refine_type)

    default_args = cfg.model.flow_refinement.copy()
    default_args.pop('type')
    default_args.update(batch_norm=cfg.model.batch_norm)

    refinement = REFINEMENTS[refine_type](**default_args)

    return refinement
