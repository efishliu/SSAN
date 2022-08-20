from QRec import QRec
from util.config import ModelConf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model', type=str, default='BPR', help='TRAIN MODEL:\n \
        1.BPR;2.SBPR;3.SERec;4.DHCF;5.DiffNet;6.LightGCN;7.MHCN;8.SEPT')
parser.add_argument('-dataset', type=str, default='Ciao', help='TRAIN DATASET:\n \
        Ciao,CiaoDVD,douban,Epinions,FilmTrust,Yelp')
parser.add_argument('-layer', type=str, default='2', help='convolution layer,default:1')
parser.add_argument('-ss_rate1', type=str, default='0.01', help='ss_rate1,default:0.005')
parser.add_argument('-ss_rate2', type=str, default='0.01', help='ss_rate2,default:0.005')
parser.add_argument('-emb_size', type=str, default='50', help='embedding size,default:50')
parser.add_argument('-epoch', type=str, default='50', help='epoch num,default:50')
parser.add_argument('-r_pos', type=str, default='3', help='rating pos,default:3')
parser.add_argument('-batch_size', type=str, default='2000', help='batch size,default:2000')
parser.add_argument('-info', type=str, default='None', help='SSAN info,default:None')
args = parser.parse_args()

if __name__ == '__main__':

    print('='*80)
    print('   QRec: An effective python-based recommendation model library.   ')
    print('='*80)

    import time
    s = time.time()
    #Register your model here and add the conf file into the config directory
    try:
        conf = ModelConf('./config/' + args.model + '.conf')
        conf.updateConf('dataset',args.dataset)
        conf.updateConf('ratings','./dataset/'+ args.dataset + '/ratings.txt')
        conf.updateConf('num.factors',args.emb_size)
        conf.updateConf('num.max.epoch',args.epoch)
        conf.updateConf('batch_size',args.batch_size)
        if conf.contains('social'):
            conf.updateConf('social','./dataset/'+ args.dataset + '/trusts.txt')
        if args.model == 'SSAN':
            conf.updateConf('SSAN','-n_layer '+ args.layer + ' -ss_rate1 ' + args.ss_rate1 + ' -ss_rate2 ' + args.ss_rate2 \
                + ' -r_pos ' + args.r_pos + ' -info ' + args.info)
        if args.model == 'SSAN2':
            conf.updateConf('SSAN2','-n_layer '+' -info ' + args.info)
        if args.model == 'MHCN':
            conf.updateConf('MHCN','-n_layer '+ args.layer + ' -ss_rate ' + args.ss_rate1) 

    except KeyError:
        print('wrong model!')
        exit(-1)
    
    recSys = QRec(conf)
    recSys.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
