from .flow import flow_output_evaluation_in_pandas
from .stereo import disp_output_evaluation_in_pandas

def output_evaluation_in_pandas(output_dict, task='stereo'):
    if task == 'stereo':
        return disp_output_evaluation_in_pandas(output_dict)
    elif task == 'flow':
        return flow_output_evaluation_in_pandas(output_dict)
    else:
        raise NotImplementedError


from .flow import remove_padding as flow_remove_padding
from .stereo import remove_padding as disp_remove_padding

def calc_error(batch, size, task='stereo'):
    if task == 'stereo':
        return disp_remove_padding(batch, size)
    elif task == 'flow':
        return flow_remove_padding(batch, size)
    else:
        raise NotImplementedError
