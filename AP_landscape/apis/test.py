import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results
#from audtorch.metrics.functional import pearsonr

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, model2, gpu_collect=False):
#def multi_gpu_test(model, model2, data_loader, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    model2.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    avg_feature_error = 0
    avg_class_error = 0
    avg_bbox_error = 0
    avg_pearson = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            x1, outs1, _ = model(return_loss=False,former_x=avg_pearson, rescale=True, **data)
            x2, outs2, result = model2(return_loss=False, former_x=x1, rescale=True, **data)

            '''
            _,_,h,w = x1[2].size()

            f5_t=x1[2].reshape(-1, h*w)
            f5_s=x2[2].reshape(-1, h*w)

            avg_pearson = avg_pearson + pearsonr(f5_s, f5_t).mean()

            
            p3_t=x1[0].permute(0, 2, 3, 1).reshape(-1, 256)
            p3_s=x2[0].permute(0, 2, 3, 1).reshape(-1, 256)
            error_feature3 = torch.abs(p3_t - p3_s).mean(1).sum()

            p4_t=x1[1].permute(0, 2, 3, 1).reshape(-1, 256)
            p4_s=x2[1].permute(0, 2, 3, 1).reshape(-1, 256)
            error_feature4 = torch.abs(p4_t - p4_s).mean(1).sum()

            p5_t=x1[2].permute(0, 2, 3, 1).reshape(-1, 256)
            p5_s=x2[2].permute(0, 2, 3, 1).reshape(-1, 256)
            error_feature5 = torch.abs(p5_t - p5_s).mean(1).sum()

            p6_t=x1[3].permute(0, 2, 3, 1).reshape(-1, 256)
            p6_s=x2[3].permute(0, 2, 3, 1).reshape(-1, 256)
            error_feature6 = torch.abs(p6_t - p6_s).mean(1).sum()

            p7_t=x1[4].permute(0, 2, 3, 1).reshape(-1, 256)
            p7_s=x2[4].permute(0, 2, 3, 1).reshape(-1, 256)
            error_feature7 = torch.abs(p7_t - p7_s).mean(1).sum()
            avg_feature_error = avg_feature_error + (error_feature3 + error_feature4 + error_feature5 + error_feature6 + error_feature7) / (p3_t.size(0) + p4_t.size(0) + p5_t.size(0) + p6_t.size(0) + p7_t.size(0))

            c3_t=outs1[0][0].permute(0, 2, 3, 1).reshape(-1, 80)
            c3_s=outs2[0][0].permute(0, 2, 3, 1).reshape(-1, 80)
            error_class3 = torch.abs(c3_t - c3_s).mean(1).sum()

            c4_t=outs1[0][1].permute(0, 2, 3, 1).reshape(-1, 80)
            c4_s=outs2[0][1].permute(0, 2, 3, 1).reshape(-1, 80)
            error_class4 = torch.abs(c4_t - c4_s).mean(1).sum()

            c5_t=outs1[0][2].permute(0, 2, 3, 1).reshape(-1, 80)
            c5_s=outs2[0][2].permute(0, 2, 3, 1).reshape(-1, 80)
            error_class5 = torch.abs(c5_t - c5_s).mean(1).sum()

            c6_t=outs1[0][3].permute(0, 2, 3, 1).reshape(-1, 80)
            c6_s=outs2[0][3].permute(0, 2, 3, 1).reshape(-1, 80)
            error_class6 = torch.abs(c6_t - c6_s).mean(1).sum()

            c7_t=outs1[0][4].permute(0, 2, 3, 1).reshape(-1, 80)
            c7_s=outs2[0][4].permute(0, 2, 3, 1).reshape(-1, 80)
            error_class7 = torch.abs(c7_t - c7_s).mean(1).sum()

            avg_class_error = avg_class_error + (error_class3 + error_class4 + error_class5 + error_class6 + error_class7) / (p3_t.size(0) + p4_t.size(0) + p5_t.size(0) + p6_t.size(0) + p7_t.size(0))

            b3_t=outs1[1][0].permute(0, 2, 3, 1).reshape(-1, 68)
            b3_s=outs2[1][0].permute(0, 2, 3, 1).reshape(-1, 68)
            error_bbox3 = torch.abs(b3_t - b3_s).mean(1).sum()

            b4_t=outs1[1][1].permute(0, 2, 3, 1).reshape(-1, 68)
            b4_s=outs2[1][1].permute(0, 2, 3, 1).reshape(-1, 68)
            error_bbox4 = torch.abs(b4_t - b4_s).mean(1).sum()

            b5_t=outs1[1][2].permute(0, 2, 3, 1).reshape(-1, 68)
            b5_s=outs2[1][2].permute(0, 2, 3, 1).reshape(-1, 68)
            error_bbox5 = torch.abs(b5_t - b5_s).mean(1).sum()

            b6_t=outs1[1][3].permute(0, 2, 3, 1).reshape(-1, 68)
            b6_s=outs2[1][3].permute(0, 2, 3, 1).reshape(-1, 68)
            error_bbox6 = torch.abs(b6_t - b6_s).mean(1).sum()

            b7_t=outs1[1][4].permute(0, 2, 3, 1).reshape(-1, 68)
            b7_s=outs2[1][4].permute(0, 2, 3, 1).reshape(-1, 68)
            error_bbox7 = torch.abs(b7_t - b7_s).mean(1).sum()

            avg_bbox_error = avg_bbox_error + (error_bbox3 + error_bbox4 + error_bbox5 + error_bbox6 + error_bbox7) / (p3_t.size(0) + p4_t.size(0) + p5_t.size(0) + p6_t.size(0) + p7_t.size(0))
            '''
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()
    #print(avg_pearson/5000)
    #print(avg_feature_error/5000)
    #print(avg_class_error/5000)
    #print(avg_bbox_error/5000)
    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        #results = collect_results_cpu(results, len(dataset), tmpdir)
        results = collect_results_cpu(results, len(dataset))
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
