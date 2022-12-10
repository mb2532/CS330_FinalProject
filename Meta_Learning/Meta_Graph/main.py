#########################################################
# Code adapted from "Link Prediction from Sparse DataUsing Meta Learning" 
# by: Avishek Joey Bose, Ankit Jain, Piero Molino, William L. Hamilton
#########################################################
import torch
import torch.nn.functional as F
import os
import sys
import os.path as osp
import argparse
from data.data import load_dataset
from torch_geometric.datasets import Planetoid,PPI,TUDataset
import torch_geometric.transforms as T
import json
from torch_geometric.nn import GATConv, GCNConv
from models.autoencoder import MyGAE, MyVGAE
from torch_geometric.data import DataLoader
from maml import meta_gradient_step
from models.models import *
from utils.utils import global_test, test, EarlyStopping, seed_everything,\
        filter_state_dict, create_nx_graph, calc_adamic_adar_score,\
        create_nx_graph_deepwalk, train_deepwalk_model,calc_deepwalk_score
from utils.utils import run_analysis
from collections import OrderedDict
from torchviz import make_dot
import numpy as np
import ipdb

def test(args,meta_model,optimizer,test_loader,train_epoch,return_val=False,inner_steps=10,seed= 0):
    ''' Meta-Testing '''
    mode='Test'
    test_graph_id_local = 0
    test_graph_id_global = 0
    args.resplit = False
    epoch=0
    args.final_test = False
    inner_test_auc_array = None
    inner_test_ap_array = None
    if return_val:
        args.inner_steps = inner_steps
        args.final_test = True
        inner_test_auc_array = np.zeros((len(test_loader)*args.test_batch_size, int(1000/5)))
        inner_test_ap_array = np.zeros((len(test_loader)*args.test_batch_size, int(1000/5)))

    meta_loss = torch.Tensor([0])
    test_avg_auc_list, test_avg_ap_list = [], []
    test_inner_avg_auc_list, test_inner_avg_ap_list = [], []
    for j,data in enumerate(test_loader):
        test_graph_id_local, meta_loss, test_inner_avg_auc_list, test_inner_avg_ap_list = meta_gradient_step(meta_model,\
                args,data,optimizer,args.inner_steps,args.inner_lr,args.order,test_graph_id_local,mode,\
                test_inner_avg_auc_list, test_inner_avg_ap_list,epoch,j,False,\
                        inner_test_auc_array,inner_test_ap_array)
        auc_list, ap_list = global_test(args,meta_model,data,OrderedDict(meta_model.named_parameters()))
        test_avg_auc_list.append(sum(auc_list)/len(auc_list))
        test_avg_ap_list.append(sum(ap_list)/len(ap_list))

        ''' Test Logging '''
    print("Failed on %d graphs" %(args.fail_counter))
    print("Epoch: %d | Test Global Avg Auc %f | Test Global Avg AP %f" \
            %(train_epoch, sum(test_avg_auc_list)/len(test_avg_auc_list),\
                    sum(test_avg_ap_list)/len(test_avg_ap_list)))
    if len(test_inner_avg_ap_list) > 0:
        print('Epoch {:01d} | Test Inner AUC: {:.4f}, AP: {:.4f}'.format(train_epoch,sum(test_inner_avg_auc_list)/len(test_inner_avg_auc_list),sum(test_inner_avg_ap_list)/len(test_inner_avg_ap_list)))

    if return_val:
        test_avg_auc = sum(test_avg_auc_list)/len(test_avg_auc_list)
        test_avg_ap = sum(test_avg_ap_list)/len(test_avg_ap_list)
        if len(test_inner_avg_ap_list) > 0:
            test_inner_avg_auc = sum(test_inner_avg_auc_list)/len(test_inner_avg_auc_list)
            test_inner_avg_ap = sum(test_inner_avg_ap_list)/len(test_inner_avg_ap_list)
        #Remove All zero rows
        test_auc_array = inner_test_auc_array[~np.all(inner_test_auc_array == 0, axis=1)]
        test_ap_array = inner_test_ap_array[~np.all(inner_test_ap_array == 0, axis=1)]
        test_aggr_auc = np.sum(test_auc_array,axis=0)/len(test_loader)
        test_aggr_ap = np.sum(test_ap_array,axis=0)/len(test_loader)
        max_auc = np.max(test_aggr_auc)
        max_ap = np.max(test_aggr_ap)
        auc_metric = 'Test_Complete' +'_AUC'
        ap_metric = 'Test_Complete' +'_AP'
        for val_idx in range(0,test_auc_array.shape[1]):
            auc = test_aggr_auc[val_idx]
            ap = test_aggr_ap[val_idx]
        print("Test Max AUC :%f | Test Max AP: %f" %(max_auc,max_ap))

        ''' Save Local final params '''
        if not os.path.exists('../saved_models/'):
            os.makedirs('../saved_models/')
        save_path = '../saved_models/' + args.namestr + '_local.pt'
        torch.save(meta_model.state_dict(), save_path)
        return max_auc, max_ap

def validation(args,meta_model,optimizer,val_loader,train_epoch,return_val=False):
    ''' Meta-Valing '''
    mode='Val'
    val_graph_id_local = 0
    val_graph_id_global = 0
    args.resplit = True
    epoch=0
    meta_loss = torch.Tensor([0])
    val_avg_auc_list, val_avg_ap_list = [], []
    val_inner_avg_auc_list, val_inner_avg_ap_list = [], []
    args.final_val = False
    inner_val_auc_array = None
    inner_val_ap_array = None
    if return_val:
        args.inner_steps = inner_steps
        args.final_val = True
        inner_val_auc_array = np.zeros((len(val_loader)*args.val_batch_size, int(1000/5)))
        inner_val_ap_array = np.zeros((len(val_loader)*args.val_batch_size, int(1000/5)))
    for j,data in enumerate(val_loader):
        val_graph_id_local, meta_loss, val_inner_avg_auc_list, val_inner_avg_ap_list = meta_gradient_step(meta_model,\
                args,data,optimizer,args.inner_steps,args.inner_lr,args.order,val_graph_id_local,mode,\
                val_inner_avg_auc_list,val_inner_avg_ap_list,epoch,j,False,\
                        inner_val_auc_array,inner_val_ap_array)
        auc_list, ap_list = global_test(args,meta_model,data,OrderedDict(meta_model.named_parameters()))
        val_avg_auc_list.append(sum(auc_list)/len(auc_list))
        val_avg_ap_list.append(sum(ap_list)/len(ap_list))

    print("Val Avg Auc %f | Val Avg AP %f" %(sum(val_avg_auc_list)/len(val_avg_auc_list),\
            sum(val_avg_ap_list)/len(val_avg_ap_list)))
    if len(val_inner_avg_ap_list) > 0:
        print("Val Inner Avg Auc %f | Val Avg AP %f" %(sum(val_inner_avg_auc_list)/len(val_inner_avg_auc_list),\
                sum(val_inner_avg_ap_list)/len(val_inner_avg_ap_list)))
    if return_val:
        val_avg_auc = sum(val_avg_auc_list)/len(val_avg_auc_list)
        val_avg_ap = sum(val_avg_ap_list)/len(val_avg_ap_list)
        val_inner_avg_auc = sum(val_inner_avg_auc_list)/len(val_inner_avg_auc_list)
        val_inner_avg_ap = sum(val_inner_avg_ap_list)/len(val_inner_avg_ap_list)
        #Remove All zero rows
        val_auc_array = inner_val_auc_array[~np.all(inner_val_auc_array == 0, axis=1)]
        val_ap_array = inner_val_ap_array[~np.all(inner_val_ap_array == 0, axis=1)]

        val_aggr_auc = np.sum(val_auc_array,axis=0)/len(val_loader)
        val_aggr_ap = np.sum(val_ap_array,axis=0)/len(val_loader)
        max_auc = np.max(val_aggr_auc)
        max_ap = np.max(val_aggr_ap)
        auc_metric = 'Val_Complete' +'_AUC'
        ap_metric = 'Val_Complete' +'_AP'
        for val_idx in range(0,val_auc_array.shape[1]):
            auc = val_aggr_auc[val_idx]
            ap = val_aggr_ap[val_idx]
        print("Val Max AUC :%f | Val Max AP: %f" %(max_auc,max_ap))
        return max_auc, max_ap

def main(args):
    assert args.model in ['GAE', 'VGAE']
    kwargs = {'GAE': MyGAE, 'VGAE': MyVGAE}
    kwargs_enc = {'GCN': MetaEncoder, 'FC': MLPEncoder, 'MLP': MetaMLPEncoder,
                  'GraphSignature': MetaSignatureEncoder,
                  'GatedGraphSignature': MetaGatedSignatureEncoder}

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
    train_loader, val_loader, test_loader = load_dataset(args.dataset,args)
    meta_model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
    if args.train_only_gs:
        trainable_parameters = []
        for name, p in meta_model.named_parameters():
            if "signature" in name:
                trainable_parameters.append(p)
            else:
                p.requires_grad = False
        optimizer = torch.optim.Adam(trainable_parameters, lr=args.meta_lr)
    else:
        optimizer = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)

    total_loss = 0
    if not args.do_kl_anneal:
        args.kl_anneal = 1

    if args.encoder == 'GraphSignature' or args.encoder == 'GatedGraphSignature':
        args.allow_unused = True
    else:
        args.allow_unused = False


    ''' Meta-training '''
    mode = 'Train'
    meta_loss = torch.Tensor([0])
    args.final_test = False
    for epoch in range(0,args.epochs):
        graph_id_local = 0
        graph_id_global = 0
        train_inner_avg_auc_list, train_inner_avg_ap_list = [], []
        if epoch > 0 and args.dataset !='PPI':
            args.resplit = False
        for i,data in enumerate(train_loader):

            graph_id_local, meta_loss, train_inner_avg_auc_list, train_inner_avg_ap_list = meta_gradient_step(meta_model,\
                    args,data,optimizer,args.inner_steps,args.inner_lr,args.order,graph_id_local,\
                    mode,train_inner_avg_auc_list, train_inner_avg_ap_list,epoch,i,True)

            auc_list, ap_list = global_test(args,meta_model,data,OrderedDict(meta_model.named_parameters()))
            graph_id_global += len(ap_list)

        if len(train_inner_avg_ap_list) > 0:
            print('Train Inner AUC: {:.4f}, AP: {:.4f}'.format(sum(train_inner_avg_auc_list)/len(train_inner_avg_auc_list),\
                            sum(train_inner_avg_ap_list)/len(train_inner_avg_ap_list)))

        ''' Meta-Testing After every Epoch'''
        meta_model_copy = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
        meta_model_copy.load_state_dict(meta_model.state_dict())
        if args.train_only_gs:
            optimizer_copy = torch.optim.Adam(trainable_parameters, lr=args.meta_lr)
        else:
            optimizer_copy = torch.optim.Adam(meta_model_copy.parameters(), lr=args.meta_lr)
        optimizer_copy.load_state_dict(optimizer.state_dict())
        validation(args,meta_model_copy,optimizer_copy,val_loader,epoch)
        test(args,meta_model_copy,optimizer_copy,test_loader,epoch,inner_steps=args.inner_steps)

    print("Failed on %d Training graphs" %(args.fail_counter))

    ''' Save Global Params '''
    if not os.path.exists('../saved_models/'):
        os.makedirs('../saved_models/')
    save_path = '../saved_models/meta_vgae.pt'
    save_path = '../saved_models/' + args.namestr + '_global_.pt'
    torch.save(meta_model.state_dict(), save_path)

    ''' Run to Convergence '''
    val_inner_avg_auc, val_inner_avg_ap = test(args,meta_model,optimizer,val_loader,epoch,\
            return_val=True,inner_steps=1000)
    test_inner_avg_auc, test_inner_avg_ap = test(args,meta_model,optimizer,test_loader,epoch,\
            return_val=True,inner_steps=1000)

    val_eval_metric = 0.5*val_inner_avg_auc + 0.5*val_inner_avg_ap
    test_eval_metric = 0.5*test_inner_avg_auc + 0.5*test_inner_avg_ap
    return val_eval_metric

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--num_channels', type=int, default='16')
    parser.add_argument('--dataset', type=str, default='CITIES')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--train_batch_size', default=4, type=int)
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--num_fixed_features', default=20, type=int)
    parser.add_argument('--num_concat_features', default=10, type=int)
    parser.add_argument('--meta_train_edge_ratio', type=float, default='0.2')
    parser.add_argument('--meta_val_edge_ratio', type=float, default='0.2')
    parser.add_argument('--k_core', type=int, default=5, help="K-core for Graph")
    parser.add_argument('--clip', type=float, default='1', help='Gradient Clip')
    parser.add_argument('--clip_weight_val', type=float, default='0.1',\
            help='Weight Clip')
    parser.add_argument('--train_ratio', type=float, default='0.8', \
            help='Used to split number of graphs for training if not provided')
    parser.add_argument('--val_ratio', type=float, default='0.1',\
            help='Used to split number of graphs for va1idation if not provided')
    parser.add_argument('--num_gated_layers', default=4, type=int,\
            help='Number of layers to use for the Gated Graph Conv Layer')
    parser.add_argument('--mlp_lr', default=1e-3, type=float)
    parser.add_argument('--inner-lr', default=0.01, type=float)
    parser.add_argument('--reset_inner_factor', default=20, type=float)
    parser.add_argument('--meta-lr', default=0.001, type=float)
    parser.add_argument('--order', default=2, type=int, help='MAML gradient order')
    parser.add_argument('--inner_steps', type=int, default=50)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--concat_fixed_feats", action="store_true", default=False,
		help='Concatenate random node features to current node features')
    parser.add_argument("--extra_backward", action="store_true", default=False,
		help='Do Extra Backward pass like in Original Pytorch MAML repo')
    parser.add_argument("--use_fixed_feats", action="store_true", default=False,
		help='Use a random node features')
    parser.add_argument("--use_same_fixed_feats", action="store_true", default=False,
		help='Use a random node features for all nodes')
    parser.add_argument('--min_nodes', type=int, default=1000, \
            help='Min Nodes needed for a graph to be included')
    parser.add_argument('--max_nodes', type=int, default=50000, \
            help='Max Nodes needed for a graph to be included')
    parser.add_argument('--kl_anneal', type=int, default=0, \
            help='KL Anneal Coefficient')
    parser.add_argument('--do_kl_anneal', action="store_true", default=False, \
            help='Do KL Annealing')
    parser.add_argument("--clip_grad", action="store_true", default=False,
		help='Gradient Clipping')
    parser.add_argument("--clip_weight", action="store_true", default=False,
		help='Weight Clipping')
    parser.add_argument("--use_gcn_sig", action="store_true", default=False,
		help='Use GCN in Signature Function')
    parser.add_argument("--train_only_gs", action="store_true", default=False,
		help='Train only the Graph Signature Function')
    parser.add_argument('--seed', type=int, default=123, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug')
    parser.add_argument('--namestr', type=str, default='Meta-Graph', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--layer_norm', default=False, action='store_true',
                        help='use layer norm')
    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name='meta-graph'

    print(vars(args))
    eval_metric = main(args)
