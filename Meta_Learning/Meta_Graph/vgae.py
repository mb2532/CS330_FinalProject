import os
import wandb
import os.path as osp
from comet_ml import Experiment
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,PPI
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE
from torch_geometric.data import DataLoader
import torch_geometric.utils as pyg_utils
import numpy as np
from data.data import load_dataset
from models import *
from utils.utils import global_test, test, EarlyStopping, seed_everything
import json
import ipdb
from models.models import Encoder, MLPEncoder, MetaMLPEncoder, MetaSignatureEncoder
import pdb
from tqdm import tqdm

def train(model, args, x, train_pos_edge_index, num_nodes, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.model in ['VGAE']:
        loss = loss + (1 / num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()

def val(model, args, x, val_pos_edge_index, num_nodes):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, val_pos_edge_index)
        loss = model.recon_loss(z, val_pos_edge_index)
        if args.model in ['VGAE']:
            loss = loss + (1 / num_nodes) * model.kl_loss()
    return loss.item()

def test(model, x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

def main(args):
    assert args.model in ['GAE', 'VGAE']
    kwargs = {'GAE': GAE, 'VGAE': VGAE}
    kwargs_enc = {'GCN': Encoder, 'FC': MLPEncoder, 'MLP': MetaMLPEncoder,
                  'GraphSignature': MetaSignatureEncoder}

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
    train_loader, val_loader, test_loader = load_dataset(args.dataset,args)
    model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_loss = 0
    graph_id = 0
    train_auc_array = np.zeros((len(train_loader)*args.train_batch_size, int(args.epochs/5 + 1)))
    train_ap_array = np.zeros((len(train_loader)*args.train_batch_size, int(args.epochs/5 + 1)))
    val_auc_array = np.zeros((len(val_loader)*args.test_batch_size, int(args.epochs/5 + 1)))
    val_ap_array = np.zeros((len(val_loader)*args.test_batch_size, int(args.epochs/5 + 1)))
    test_auc_array = np.zeros((len(test_loader)*args.test_batch_size, int(args.epochs/5 + 1)))
    test_ap_array = np.zeros((len(test_loader)*args.test_batch_size, int(args.epochs/5 + 1)))

    print("finetuning")
    if args.finetune:
        for i,data_batch in enumerate(train_loader):
            for idx, data in enumerate(data_batch):
                # temp_x = torch.from_numpy(np.asarray(list(range(data.x.size(0)))))
                # data.x = temp_x.unsqueeze(1)
                # data.y = None
                # data.x = torch.cat((data.x, data.x[0, :].unsqueeze(0)), dim=0)
                data.train_mask = data.val_mask = data.test_mask = data.y = None
                data.batch = None
                num_nodes = data.num_nodes
                meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
                if args.use_fixed_feats:
                    ##TODO: Should this be a fixed embedding table instead of generating this each time?
                    perm = torch.randperm(args.feats.size(0))
                    perm_idx = perm[:num_nodes]
                    data.x = args.feats[perm_idx]

                if args.concat_fixed_feats:
                    ##TODO: Should this be a fixed embedding table instead of generating this each time?
                    concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                    data.x = torch.cat((data.x,concat_feats),1)
               
                try:
                    data = pyg_utils.train_test_split_edges(data,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
                except:
                    args.fail_counter += 1
                    print("5Failed on Graph %d" %(graph_id))
                    continue

                # Additional Failure Checks for small graphs
                if data.val_pos_edge_index.size()[1] == 0 or data.test_pos_edge_index.size()[1] == 0:
                    args.fail_counter += 1
                    print("6Failed on Graph %d" %(graph_id))
                    continue

                x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
                for epoch in tqdm(range(0, args.epochs)):
                    if not args.random_baseline:
                        if args.train_with_val:
                            train_pos_edge_index = torch.cat([data.train_pos_edge_index,
                                       data.val_pos_edge_index], dim=1).cuda()
                        train(model,args,x,train_pos_edge_index,data.num_nodes,optimizer)
                    auc, ap = test(model, x, train_pos_edge_index,
                            data.test_pos_edge_index, data.test_neg_edge_index)

                    if epoch % 5 == 0:
                        my_step = int(epoch / 5)
                        train_auc_array[graph_id][my_step] = auc
                        train_ap_array[graph_id][my_step] = ap

                ''' Save after every graph '''
                auc_metric = 'Train_Local_Batch_Graph_' + str(graph_id) +'_AUC'
                ap_metric = 'Train_Local_Batch_Graph_' + str(graph_id) +'_AP'
                for val_idx in range(0,train_auc_array.shape[1]):
                    auc = train_auc_array[graph_id][val_idx]
                    ap = train_ap_array[graph_id][val_idx]
                auc = train_auc_array[graph_id][val_idx - 1]
                ap = train_ap_array[graph_id][val_idx - 1]
                print('Train Graph {:01d}, AUC: {:.4f}, AP: {:.4f}'.format(graph_id, auc, ap))
                graph_id += 1
                save_path = '../saved_models/vgae.pt'
                # torch.save(model.state_dict(), save_path)

        auc_metric = 'Train_Complete' +'_AUC'
        ap_metric = 'Train_Complete' +'_AP'
        #Remove All zero rows
        train_auc_array = train_auc_array[~np.all(train_auc_array == 0, axis=1)]
        train_ap_array = train_ap_array[~np.all(train_ap_array == 0, axis=1)]
        train_aggr_auc = np.sum(train_auc_array,axis=0)/len(train_loader)
        train_aggr_ap = np.sum(train_ap_array,axis=0)/len(train_loader)
        for val_idx in range(0,train_auc_array.shape[1]):
            auc = train_aggr_auc[val_idx]
            ap = train_aggr_ap[val_idx]

    ''' Start Validation '''
    if args.do_val:
        val_graph_id = 0
        for i,data_batch in enumerate(val_loader):
            ''' Re-init Optimizerr '''
            if args.finetune:
                val_model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
                val_model.load_state_dict(model.state_dict())
                optimizer = torch.optim.Adam(val_model.parameters(), lr=args.lr)
            else:
                val_model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
                optimizer = torch.optim.Adam(val_model.parameters(), lr=args.lr)
            early_stopping = EarlyStopping(patience=args.patience, verbose=False)

            for idx, data in enumerate(data_batch):
                data.train_mask = data.val_mask = data.test_mask = data.y = None
                data.batch = None
                num_nodes = data.num_nodes
                if args.use_fixed_feats:
                    ##TODO: Should this be a fixed embedding table instead of generating this each time?
                    perm = torch.randperm(args.feats.size(0))
                    perm_idx = perm[:num_nodes]
                    data.x = args.feats[perm_idx]

                if args.concat_fixed_feats:
                    ##TODO: Should this be a fixed embedding table instead of generating this each time?
                    concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                    data.x = torch.cat((data.x,concat_feats),1)

                # Val Ratio is Fixed at 0.1
                meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
                try:
                    data = pyg_utils.train_test_split_edges(data,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
                except:
                    args.fail_counter += 1
                    print("1Failed on Graph %d" %(val_graph_id))
                    continue

                # Additional Failure Checks for small graphs
                if data.val_pos_edge_index.size()[1] == 0 or data.test_pos_edge_index.size()[1] == 0:
                    args.fail_counter += 1
                    print("2Failed on Graph %d" %(val_graph_id))
                    continue

                x, train_pos_edge_index = data.x.to(args.dev), data.train_pos_edge_index.to(args.dev)
                val_pos_edge_index = data.val_pos_edge_index.to(args.dev)
                for epoch in range(0, args.epochs):
                    if not args.random_baseline:
                        train(val_model,args,x,train_pos_edge_index,data.num_nodes,optimizer)
                        val_loss = val(model,args,x,val_pos_edge_index,data.num_nodes)
                        early_stopping(val_loss, val_model)
                    auc, ap = test(val_model, x, train_pos_edge_index,
                            data.test_pos_edge_index, data.test_neg_edge_index)

                    if early_stopping.early_stop:
                        print("Early stopping for Graph %d | AUC: %f AP: %f" \
                                %(val_graph_id, auc, ap))
                        my_step = int(epoch / 5)
                        val_auc_array[val_graph_id][my_step:,] = auc
                        val_ap_array[val_graph_id][my_step:,] = ap
                        break

                    if epoch % 5 == 0:
                        my_step = int(epoch / 5)
                        val_auc_array[val_graph_id][my_step] = auc
                        val_ap_array[val_graph_id][my_step] = ap

                ''' Save after every graph '''
                auc_metric = 'Val_Local_Batch_Graph_' + str(val_graph_id) +'_AUC'
                ap_metric = 'Val_Local_Batch_Graph_' + str(val_graph_id) +'_AP'
                for val_idx in range(0,val_auc_array.shape[1]):
                    auc = val_auc_array[val_graph_id][val_idx]
                    ap = val_ap_array[val_graph_id][val_idx]

                print('Val Graph {:01d}, AUC: {:.4f}, AP: {:.4f}'.format(val_graph_id, auc, ap))
                val_graph_id += 1

        auc_metric = 'Val_Complete' +'_AUC'
        ap_metric = 'Val_Complete' +'_AP'

        #Remove All zero rows
        val_auc_array = val_auc_array[~np.all(val_auc_array == 0, axis=1)]
        val_ap_array = val_ap_array[~np.all(val_ap_array == 0, axis=1)]
        val_aggr_auc = np.sum(val_auc_array,axis=0)/len(val_loader)
        val_aggr_ap = np.sum(val_ap_array,axis=0)/len(val_loader)
        max_auc = np.max(val_aggr_auc)
        max_ap = np.max(val_aggr_ap)
        for val_idx in range(0,val_auc_array.shape[1]):
            auc = val_aggr_auc[val_idx]
            ap = val_aggr_ap[val_idx]
        auc = val_aggr_auc[val_idx -1]
        ap = val_aggr_ap[val_idx - 1]
        print('Val Complete AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
        print('Val Max AUC: {:.4f}, AP: {:.4f}'.format(max_auc, max_ap))
        val_eval_metric = 0.5*max_auc + 0.5*max_ap
        return val_eval_metric

    ''' Start Testing '''
    print("start testing")
    if not args.do_val:
        test_graph_id = 0
        for i,data_batch in enumerate(test_loader):
            ''' Re-init Optimizerr '''
            if args.finetune:
                test_model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
                test_model.load_state_dict(model.state_dict())
                optimizer = torch.optim.Adam(test_model.parameters(), lr=args.lr)
            else:
                test_model = kwargs[args.model](kwargs_enc[args.encoder](args, args.num_features, args.num_channels)).to(args.dev)
                optimizer = torch.optim.Adam(test_model.parameters(), lr=args.lr)
            early_stopping = EarlyStopping(patience=args.patience, verbose=False)

            for idx, data in enumerate(data_batch):
                data.train_mask = data.val_mask = data.test_mask = data.y = None
                data.batch = None
                num_nodes = data.num_nodes
                if args.use_fixed_feats:
                    ##TODO: Should this be a fixed embedding table instead of generating this each time?
                    perm = torch.randperm(args.feats.size(0))
                    perm_idx = perm[:num_nodes]
                    data.x = args.feats[perm_idx]

                if args.concat_fixed_feats:
                    ##TODO: Should this be a fixed embedding table instead of generating this each time?
                    concat_feats = torch.randn(num_nodes,args.num_concat_features,requires_grad=False)
                    data.x = torch.cat((data.x,concat_feats),1)

                # Val Ratio is Fixed at 0.1
                meta_test_edge_ratio = 1 - args.meta_val_edge_ratio - args.meta_train_edge_ratio
                data = pyg_utils.train_test_split_edges(data,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
                # try:
                #     data = pyg_utils.train_test_split_edges(data,val_ratio=args.meta_val_edge_ratio,test_ratio=meta_test_edge_ratio)
                # except:
                #     args.fail_counter += 1
                #     print("3Failed on Graph %d" %(test_graph_id))
                #     continue

                # Additional Failure Checks for small graphs
                if data.val_pos_edge_index.size()[1] == 0 or data.test_pos_edge_index.size()[1] == 0:
                    args.fail_counter += 1
                    print("4Failed on Graph %d" %(test_graph_id))
                    continue
                
                x = data.x.to(args.dev)
                train_pos_edge_index = data.train_pos_edge_index.to(args.dev)
                val_pos_edge_index = data.val_pos_edge_index.to(args.dev)
                for epoch in tqdm(range(0, args.epochs)):
                    if not args.random_baseline:
                        if args.train_with_val:
                            train_pos_edge_index =torch.cat([data.train_pos_edge_index,
                                       data.val_pos_edge_index], dim=1).cuda()
                            val_loss = val(model,args,x,train_pos_edge_index,data.num_nodes)
                            early_stopping(val_loss, test_model)
                        else:
                            val_loss = val(model,args,x,val_pos_edge_index,data.num_nodes)
                            early_stopping(val_loss, test_model)
                        train(test_model,args,x,train_pos_edge_index,data.num_nodes,optimizer)
                    auc, ap = test(test_model, x, train_pos_edge_index,
                            data.test_pos_edge_index, data.test_neg_edge_index)

                    if early_stopping.early_stop:
                        print("Early stopping for Graph %d | AUC: %f AP: %f" \
                                %(test_graph_id, auc, ap))
                        my_step = int(epoch / 5)
                        test_auc_array[test_graph_id][my_step:,] = auc
                        test_ap_array[test_graph_id][my_step:,] = ap
                        break

                    if epoch % 5 == 0:
                        my_step = int(epoch / 5)
                        test_auc_array[test_graph_id][my_step] = auc
                        test_ap_array[test_graph_id][my_step] = ap

                ''' Save after every graph '''
                auc_metric = 'Test_Local_Batch_Graph_' + str(test_graph_id) +'_AUC'
                ap_metric = 'Test_Local_Batch_Graph_' + str(test_graph_id) +'_AP'
                for val_idx in range(0,test_auc_array.shape[1]):
                    auc = test_auc_array[test_graph_id][val_idx]
                    ap = test_ap_array[test_graph_id][val_idx]
                print('Test Graph {:01d}, AUC: {:.4f}, AP: {:.4f}'.format(test_graph_id, auc, ap))
                test_graph_id += 1
                if not os.path.exists('../saved_models/'):
                    os.makedirs('../saved_models/')
                save_path = '../saved_models/vgae.pt'
                # torch.save(model.state_dict(), save_path)

        auc_metric = 'Test_Complete' +'_AUC'
        ap_metric = 'Test_Complete' +'_AP'
        #Remove All zero rows
        test_auc_array = test_auc_array[~np.all(test_auc_array == 0, axis=1)]
        test_ap_array = test_ap_array[~np.all(test_ap_array == 0, axis=1)]

        test_aggr_auc = np.sum(test_auc_array,axis=0)/len(test_loader)
        test_aggr_ap = np.sum(test_ap_array,axis=0)/len(test_loader)
        max_auc = np.max(test_aggr_auc)
        max_ap = np.max(test_aggr_ap)
        for val_idx in range(0,test_auc_array.shape[1]):
            auc = test_aggr_auc[val_idx]
            ap = test_aggr_ap[val_idx]
        auc = test_aggr_auc[val_idx -1]
        ap = test_aggr_ap[val_idx -1]
        print('Test Complete AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
        print('Test Max AUC: {:.4f}, AP: {:.4f}'.format(max_auc, max_ap))
        test_eval_metric = 0.5*max_auc + 0.5*max_ap
        return test_eval_metric

if __name__ == '__main__':
    """
    Process command-line arguments, then call main()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='VGAE')
    parser.add_argument('--num_channels', type=int, default='16')
    parser.add_argument('--epochs', default=251, type=int)
    parser.add_argument('--dataset', type=str, default='CITIES')
    parser.add_argument("--finetune", action="store_true", default=False,
		help='Finetune from previous graph')
    parser.add_argument('--train_batch_size', default=1, type=int)
    parser.add_argument('--num_gated_layers', default=4, type=int,\
            help='Number of layers to use for the Gated Graph Conv Layer')
    parser.add_argument('--encoder', type=str, default='GCN')
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--num_fixed_features', default=20, type=int)
    parser.add_argument('--num_concat_features', default=10, type=int)
    parser.add_argument('--meta_train_edge_ratio', type=float, default='0.2')
    parser.add_argument('--meta_val_edge_ratio', type=float, default='0.2')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--train_ratio', type=float, default='0.8', \
            help='Used to split number of graphs for training if not provided')
    parser.add_argument("--concat_fixed_feats", action="store_true", default=False,
		help='Concatenate random node features to current node features')
    parser.add_argument("--use_fixed_feats", action="store_true", default=False,
		help='Use a random node features')
    parser.add_argument('--val_ratio', type=float, default='0.1',\
            help='Used to split number of graphs for validation if not provided')
    parser.add_argument('--train_with_val', default=False, action='store_true',
                        help='Combine Train + Val edges')
    parser.add_argument('--do_val', default=False, action='store_true',
                        help='Do Validation')
    parser.add_argument('--seed', type=int, default=12345, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Debug')
    parser.add_argument('--model_path', type=str, default="mnist_cnn.pt",
                        help='where to save/load')
    parser.add_argument("--random_baseline", action="store_true", default=False,
		help='Use a Random Baseline')
    parser.add_argument('--k_core', type=int, default=5, help="K-core for Graph")
    parser.add_argument('--patience', type=int, default=200, help="K-core for Graph")
    parser.add_argument("--reprocess", action="store_true", default=False,
		help='Reprocess AMINER datasete')
    parser.add_argument("--ego", action="store_true", default=False,
		help='Reprocess AMINER as ego dataset')
    parser.add_argument('--opus', default=False, action='store_true',
                        help='Change AMINER File Path for Opus')
    parser.add_argument('--max_nodes', type=int, default=50000, \
            help='Max Nodes needed for a graph to be included')
    parser.add_argument('--namestr', type=str, default='Meta-Graph', \
            help='additional info in output filename to describe experiments')
    parser.add_argument('--min_nodes', type=int, default=1000, \
            help='Min Nodes needed for a graph to be included')
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)


    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.dataset=='PPI':
        project_name = 'meta-graph-ppi'
    elif args.dataset == 'CITIES':
        project_name = 'meta-graph-cities'
    elif args.dataset=='AMINER':
        project_name = "meta-graph-aminer"
    else:
        project_name='meta-graph'

    print(vars(args))
    eval_metric = main(args)
