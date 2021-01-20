from mmcv.runner import HOOKS, Hook
from mmcv.runner import save_checkpoint


@HOOKS.register_module()
class EpochHook(Hook):

    def __init__(self, remove_teacher=11):
        self.remove_teacher = remove_teacher
        pass

    # def before_run(self, runner):
    #     print(runner.model.module)

    def after_run(self, runner):
        model = runner.model.module

        model.bbox_head.teacher_model = None
        save_checkpoint(model, 'final.pth')

        # def before_epoch(self, runner):
        #     pass

        # def after_epoch(self, runner):
        #     pass

        # def before_iter(self, runner):
        #     pass

        # def after_iter(self, runner):
        #     pass
