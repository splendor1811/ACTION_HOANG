import torch
from .pipeline.sampling import UniformSampleFrames, SampleFrames
from .pipeline.pose_related import PoseDecode
from .pipeline.augmentations import PoseCompact, Resize, RandomResizedCrop, Flip
from .pipeline.heatmap_related import GeneratePoseTarget
from .pipeline.formatting import FormatShape, Collect, ToTensor

class PreprocessingPoseFrames():
    def __init__(self, pipeline, test_mode):
        self.test_mode = test_mode
        if self.test_mode:
            self.uniform_sample_frames = UniformSampleFrames(clip_len=pipeline['UniformSampleFrames']['clip_len'], num_clips= pipeline['UniformSampleFrames']['num_clips'])
        else:
            self.uniform_sample_frames = UniformSampleFrames(clip_len=pipeline['UniformSampleFrames']['clip_len'])
        self.pose_decode = PoseDecode()
        self.pose_compact = PoseCompact(hw_ratio=pipeline['PoseCompact']['hw_ratio'],allow_imgpad=pipeline['PoseCompact']['allow_imgpad'])
        if self.test_mode:
            self.resize = Resize(scale=tuple(pipeline['Resize']['scale']))
        else:
            self.resize1 = Resize(scale=tuple(pipeline['Resize1']['scale']))
            self.random_crop_resize = RandomResizedCrop(area_range=tuple(pipeline['RandomResizedCrop']['area_range']))
            self.resize2 = Resize(scale=tuple(pipeline['Resize2']['scale']), keep_ratio=pipeline['Resize2']['keep_ratio'])
            self.flip = Flip(flip_ratio=pipeline['Flip']['flip_ratio'],left_kp=pipeline['Flip']['left_kp'],right_kp=pipeline['Flip']['right_kp'])
        self.generate_pose = GeneratePoseTarget(with_kp=pipeline['GeneratePoseTarget']['with_kp'],with_limb=pipeline['GeneratePoseTarget']['with_limb'])
        self.format_shape = FormatShape(input_format=pipeline['FormatShape']['input_format'])
        self.collect = Collect(keys=pipeline['Collect']['keys'], meta_keys=pipeline['Collect']['meta_keys'])
        self.to_tensor = ToTensor(keys=pipeline['ToTensor']['keys'])

    def __call__(self, result):
        res = self.uniform_sample_frames(results=result)
        res = self.pose_decode(results=res)
        res = self.pose_compact(results=res)
        if self.test_mode:
            res = self.resize(results=res)
        else:
            res = self.resize1(results=res)
            res = self.random_crop_resize(results=res)
            res = self.resize2(results=res)
            res = self.flip(results=res)
        res = self.generate_pose(results=res)
        res = self.format_shape(results=res)
        res = self.collect(results=res)
        res = self.to_tensor(results=res)

        return res

