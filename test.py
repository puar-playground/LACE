import os
from fid.model import load_fidnet_v3
from util.metric import compute_generative_model_scores, compute_maximum_iou, compute_overlap, compute_alignment
import pickle as pk
from tqdm import tqdm
from util.datasets.load_data import init_dataset
from util.visualization import save_image
from util.constraint import *
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('TkAgg')
from util.seq_util import sparse_to_dense, loader_to_list, pad_until
import argparse
from model_diffusion import Diffusion


def test_fid_feat(dataset_name, device='cuda', batch_size=20):

    if os.path.exists(f'./fid/feature/fid_feat_test_{dataset_name}.pk'):
        feats_test = pk.load(open(f'./fid/feature/fid_feat_test_{dataset_name}.pk', 'rb'))
        return feats_test

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = []

    with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'Get feature for FID',
              ncols=200) as pbar:

        for i, data in pbar:

            bbox, label, _, mask = sparse_to_dense(data)
            label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)
            padding_mask = ~mask

            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            feats_test.append(feat.detach().cpu())

    pk.dump(feats_test, open(f'./fid/feature/fid_feat_test_{dataset_name}.pk', 'wb'))

    return feats_test


def test_layout_uncond(model, batch_size=128, dataset_name='publaynet', test_plot=False,
                       save_dir='./plot/test', beautify=False):

    model.eval()
    device = model.device
    n_batch_dict = {'publaynet': int(44 * 256 / batch_size), 'rico13': int(17 * 256 / batch_size),
                    'rico25': int(17 * 256 / batch_size), 'magazine': int(512 / batch_size),
                    'crello': int(2560 / batch_size)}
    n_batch = n_batch_dict[dataset_name]

    # prepare dataset
    main_dataset, _ = init_dataset(dataset_name, './datasets', batch_size=batch_size, split='test')

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0
    with torch.no_grad():
        for i in tqdm(range(n_batch), desc='uncond testing', ncols=200, total=n_batch):
            bbox_generated, label, mask = model.reverse_ddim(batch_size=batch_size, stochastic=True, save_inter=False)
            if beautify:
                bbox_generated, mask = post_process(bbox_generated, mask)
            padding_mask = ~mask

            label[mask == False] = 0

            if torch.isnan(bbox_generated[0, 0, 0]):
                print('not a number error')
                return None

            # accumulate align and overlap
            align_norm = compute_alignment(bbox_generated, mask)
            align_sum += torch.mean(align_norm)
            overlap_score = compute_overlap(bbox_generated, mask)
            overlap_sum += torch.mean(overlap_score)


            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox_generated, label, padding_mask)
            feats_generate.append(feat.cpu())

            if test_plot and i <= 10:
                img = save_image(bbox_generated[:9], label[:9], mask[:9], draw_label=False, dataset=dataset_name)
                plt.figure(figsize=[12, 12])
                plt.imshow(img)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{dataset_name}_{i}.png'))
                # plt.close()

    result = compute_generative_model_scores(feats_test, feats_generate)
    fid = result['fid']

    align_final = 100 * align_sum / n_batch
    overlap_final = 100 * overlap_sum / n_batch

    print(f'uncond, align: {align_final}, fid: {fid}, overlap: {overlap_final}')

    return align_final, fid, overlap_final


def test_layout_cond(model, batch_size=256, cond='c', dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test', beautify=False):

    assert cond in {'c', 'cwh', 'complete'}
    model.eval()
    device = model.device

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    layout_generated = []

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0
    with torch.no_grad():

        with tqdm(enumerate(main_dataloader), total=len(main_dataloader), desc=f'cond: {cond} generation',
                  ncols=200) as pbar:

            for i, data in pbar:

                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)

                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                # shift to center
                bbox_in = 2 * (bbox - 0.5).to(device)

                # set mask to label 5
                label[mask == False] = seq_dim - 5

                label_oh = torch.nn.functional.one_hot(label, num_classes=seq_dim - 4).to(device)
                real_layout = torch.cat((label_oh, bbox_in), dim=2).to(device)

                bbox_generated, label_generated, mask_generated = model.conditional_reverse_ddim(real_layout, cond=cond)
                if beautify:
                    bbox_generated, mask_generated = post_process(bbox_generated, mask_generated)
                padding_mask = ~mask_generated

                # test for errors
                if torch.isnan(bbox[0, 0, 0]):
                    print('not a number error')
                    return None

                # accumulate align and overlap
                align_norm = compute_alignment(bbox_generated, mask)
                align_sum += torch.mean(align_norm)
                overlap_score = compute_overlap(bbox, mask)
                overlap_sum += torch.mean(overlap_score)

                # record for max_iou
                label_generated[label_generated == seq_dim - 5] = 0
                for j in range(bbox.shape[0]):
                    mask_single = mask_generated[j, :]
                    bbox_single = bbox_generated[j, mask_single, :]
                    label_single = label_generated[j, mask_single]

                    layout_generated.append((bbox_single.to('cpu').numpy(), label_single.to('cpu').numpy()))

                # record for FID
                with torch.set_grad_enabled(False):
                    feat = fid_model.extract_features(bbox_generated, label_generated, padding_mask)
                feats_generate.append(feat.cpu())

                if test_plot and i <= 10:
                    img = save_image(bbox_generated[:9], label_generated[:9], mask_generated[:9],
                                     draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(f'./plot/test/cond_{cond}_{dataset_name}_{i}.png')
                    plt.close()

                    img = save_image(bbox[:9], label[:9], mask[:9], draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'{dataset_name}_real.png'))
                    plt.close()

    maxiou = compute_maximum_iou(layouts_main, layout_generated)
    result = compute_generative_model_scores(feats_test, feats_generate)
    fid = result['fid']

    align_final = 100 * align_sum / len(main_dataloader)
    overlap_final = 100 * overlap_sum / len(main_dataloader)

    print(f'cond {cond}, align: {align_final}, fid: {fid}, maxiou: {maxiou}, overlap: {overlap_final}')

    return align_final, fid, maxiou, overlap_final


def test_layout_refine(model, batch_size=256, dataset_name='publaynet', seq_dim=10,
                     test_plot=False, save_dir='./plot/test', beautify=False):

    model.eval()
    device = model.device
    n_batch_dict = {'publaynet': 44, 'rico13': 17, 'rico25': 17, 'magazine': 2, 'crello': 10}
    n_batch = n_batch_dict[dataset_name]

    # prepare dataset
    main_dataset, main_dataloader = init_dataset(dataset_name, './datasets', batch_size=batch_size,
                                                 split='test', shuffle=False, transform=None)

    layouts_main = loader_to_list(main_dataloader)
    layout_generated = []

    fid_model = load_fidnet_v3(main_dataset, './fid/FIDNetV3', device=device)
    feats_test = test_fid_feat(dataset_name, device=device, batch_size=20)
    feats_generate = []

    align_sum = 0
    overlap_sum = 0
    with torch.no_grad():

        with tqdm(enumerate(main_dataloader), total=min(n_batch, len(main_dataloader)), desc=f'refine generation',
                  ncols=200) as pbar:

            for i, data in pbar:
                if i == min(n_batch, len(main_dataloader)):
                    break

                bbox, label, _, mask = sparse_to_dense(data)
                label, bbox, mask = pad_until(label, bbox, mask, max_seq_length=25)

                label, bbox, mask = label.to(device), bbox.to(device), mask.to(device)

                # shift to center
                bbox_noisy = torch.clamp(bbox + 0.1 * torch.randn_like(bbox), min=0, max=1)
                bbox_in_noisy = 2 * (bbox_noisy - 0.5).to(device)
                #
                # set mask to label 5
                label[mask == False] = seq_dim - 5

                label_oh = torch.nn.functional.one_hot(label, num_classes=seq_dim - 4).to(device)
                noisy_layout = torch.cat((label_oh, bbox_in_noisy), dim=2).to(device)

                bbox_refined, _, _ = model.refinement_reverse_ddim(noisy_layout)
                if beautify:
                    bbox_refined, mask = post_process(bbox_refined, mask)
                padding_mask = ~mask

                # accumulate align and overlap
                align_norm = compute_alignment(bbox_refined, mask)
                align_sum += torch.mean(align_norm)
                overlap_score = compute_overlap(bbox_refined, mask)
                overlap_sum += torch.mean(overlap_score)

                # record for max_iou
                label[label == seq_dim - 5] = 0

                for j in range(bbox_refined.shape[0]):
                    mask_single = mask[j, :]
                    bbox_single = bbox_refined[j, mask_single, :]
                    label_single = label[j, mask_single]

                    layout_generated.append((bbox_single.to('cpu').numpy(), label_single.to('cpu').numpy()))

                # record for FID
                with torch.set_grad_enabled(False):
                    feat = fid_model.extract_features(bbox_refined, label, padding_mask)
                feats_generate.append(feat.cpu())


                if test_plot and i <= 10:
                    img = save_image(bbox_refined[:9], label[:9], mask[:9],
                                     draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(f'./plot/test/refine_{dataset_name}_{i}.png')
                    plt.close()

                    img = save_image(bbox[:9], label[:9], mask[:9], draw_label=False, dataset=dataset_name)
                    plt.figure(figsize=[12, 12])
                    plt.imshow(img)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'{dataset_name}_real.png'))
                    plt.close()

    maxiou = compute_maximum_iou(layouts_main, layout_generated)
    result = compute_generative_model_scores(feats_test, feats_generate)
    fid = result['fid']

    align_final = 100 * align_sum / len(main_dataloader)
    overlap_final = 100 * overlap_sum / len(main_dataloader)

    print(f'refine, align: {align_final}, fid: {fid}, maxiou: {maxiou}, overlap: {overlap_final}')
    return align_final, fid, maxiou, overlap_final


def test_all(model, dataset_name='publaynet', seq_dim=10, test_plot=False, save_dir='./plot/test', batch_size=256,
             beautify=False):

    align_uncond, fid_uncond, overlap_uncond = test_layout_uncond(model, batch_size=batch_size, dataset_name=dataset_name,
                                                  test_plot=test_plot, save_dir=save_dir, beautify=beautify)
    align_c, fid_c, maxiou_c, overlap_c = test_layout_cond(model, batch_size=batch_size, cond='c',
                                                dataset_name=dataset_name, seq_dim=seq_dim,
                                                test_plot=test_plot, save_dir=save_dir, beautify=beautify)
    align_cwh, fid_cwh, maxiou_cwh, overlap_cwh = test_layout_cond(model, batch_size=batch_size, cond='cwh',
                                                      dataset_name=dataset_name, seq_dim=seq_dim,
                                                      test_plot=test_plot, save_dir=save_dir, beautify=beautify)
    align_complete, fid_complete, maxiou_complete, overlap_complete = test_layout_cond(model, batch_size=batch_size,
                                                                     cond='complete', dataset_name=dataset_name,
                                                                     seq_dim=seq_dim, test_plot=test_plot,
                                                                     save_dir=save_dir, beautify=beautify)
    align_r, fid_r, maxiou_r, overlap_r = test_layout_refine(model, batch_size=batch_size,
                                            dataset_name=dataset_name, seq_dim=seq_dim,
                                            test_plot=test_plot, save_dir=save_dir, beautify=beautify)

    # fid_total = fid_uncond + fid_c + fid_cwh + fid_complete
    # print(f'total fid: {fid_total}')
    # return fid_total


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=256, help="batch_size", type=int)
    parser.add_argument("--device", default='cpu', help="which GPU to use", type=str)
    parser.add_argument("--dataset", default='publaynet',
                        help="choose from [publaynet, rico13, rico25]", type=str)
    parser.add_argument("--data_dir", default='./datasets', help="dir of datasets", type=str)
    parser.add_argument("--num_workers", default=4, help="num_workers", type=int)
    parser.add_argument("--feature_dim", default=2048, help="feature_dim", type=int)
    parser.add_argument("--dim_transformer", default=1024, help="dim_transformer", type=int)
    parser.add_argument("--nhead", default=16, help="nhead attention", type=int)
    parser.add_argument("--nlayer", default=4, help="nlayer", type=int)
    parser.add_argument("--experiment", default='c', help="experiment setting [uncond, c, cwh, complete, all]", type=str)
    parser.add_argument('--plot', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--beautify', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--plot_save_dir", default='./plot/test', help="dir to save generated plot of layouts", type=str)
    args = parser.parse_args()

    # prepare data
    train_dataset, train_loader = init_dataset(args.dataset, args.data_dir, batch_size=args.batch_size,
                                               split='train', shuffle=True)
    num_class = train_dataset.num_classes + 1

    # set up model
    model_ddpm = Diffusion(num_timesteps=1000, nhead=args.nhead, dim_transformer=args.dim_transformer,
                           feature_dim=args.feature_dim, seq_dim=num_class + 4, num_layers=args.nlayer,
                           device=args.device, ddim_num_steps=100)

    state_dict = torch.load(f'./model/{args.dataset}_best.pt', map_location='cpu')
    model_ddpm.load_diffusion_net(state_dict)

    if args.experiment == 'uncond':
        test_layout_uncond(model_ddpm, batch_size=args.batch_size,
                                                      dataset_name=args.dataset, test_plot=args.plot,
                                                      save_dir=args.plot_save_dir, beautify=args.beautify)
    elif args.experiment in ['c', 'cwh', 'complete']:
         test_layout_cond(model_ddpm, batch_size=args.batch_size, cond=args.experiment,
                                              dataset_name=args.dataset, seq_dim=num_class + 4,
                                              test_plot=args.plot, save_dir=args.plot_save_dir, beautify=args.beautify)
    elif args.experiment == 'refine':
        test_layout_refine(model_ddpm, batch_size=args.batch_size,
                                                dataset_name=args.dataset, seq_dim=num_class + 4,
                                                test_plot=args.plot, save_dir=args.plot_save_dir, beautify=args.beautify)
    elif args.experiment == 'all':
        test_all(model_ddpm, dataset_name=args.dataset, seq_dim=num_class + 4, test_plot=args.plot,
                 save_dir=args.plot_save_dir, batch_size=args.batch_size, beautify=args.beautify)
    else:
        raise Exception('experiment setting undefined')




