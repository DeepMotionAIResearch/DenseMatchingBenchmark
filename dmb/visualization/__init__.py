from .flow import SaveResultTool as FlowSaveResultTool
from .stereo import SaveResultTool as DispSaveResultTool

def SaveResultTool(task):
    if task == 'stereo':
        return DispSaveResultTool()
    elif task == 'flow':
        return FlowSaveResultTool()
    else:
        raise NotImplementedError


from .flow import ShowResultTool as FlowShowResultTool
from .stereo import ShowResultTool as DispShowResultTool

def ShowResultTool(task):
    if task == 'stereo':
        return DispShowResultTool()
    elif task == 'flow':
        return FlowShowResultTool()
    else:
        raise NotImplementedError
