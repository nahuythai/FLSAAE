from struct import pack
import torch
import torch.nn as nn

import torch.utils.data as data
import numpy as np
from sklearn import preprocessing
from torch.autograd import Variable
import argparse
import pandas as pd
from io import StringIO
import time
SLEEP_INTERVAL = 1.0

parser = argparse.ArgumentParser("semi-supervised aae model")
parser.add_argument("--encoder_path", type=str, default="encoder", help="encoder path")
parser.add_argument("--data_path", type=str, default="flows.csv", help="data path")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")

args = parser.parse_args()
print(args)


cuda =False

ORIGINAL_COLUMNS = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'src_mac', 'dst_mac',
       'protocol', 'timestamp', 'flow_duration', 'flow_byts_s', 'flow_pkts_s',
       'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts', 'tot_bwd_pkts',
       'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max',
       'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
       'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean',
       'bwd_pkt_len_std', 'pkt_len_max', 'pkt_len_min', 'pkt_len_mean',
       'pkt_len_std', 'pkt_len_var', 'fwd_header_len', 'bwd_header_len',
       'fwd_seg_size_min', 'fwd_act_data_pkts', 'flow_iat_mean',
       'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot',
       'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std',
       'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean',
       'bwd_iat_std', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags',
       'bwd_urg_flags', 'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt',
       'psh_flag_cnt', 'ack_flag_cnt', 'urg_flag_cnt', 'ece_flag_cnt',
       'down_up_ratio', 'pkt_size_avg', 'init_fwd_win_byts',
       'init_bwd_win_byts', 'active_max', 'active_min', 'active_mean',
       'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std',
       'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg',
       'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg',
       'bwd_seg_size_avg', 'cwe_flag_count', 'subflow_fwd_pkts',
       'subflow_bwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_byts']

INSECLAB_PREPROCESSED_COLUMNS = ['dst_port', 'protocol', 'flow_duration', 'flow_byts_s', 'flow_pkts_s', 'fwd_pkts_s', 'bwd_pkts_s', 'tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max', 'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std', 'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean', 'bwd_pkt_len_std', 'pkt_len_max', 'pkt_len_min', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var', 'fwd_header_len', 'bwd_header_len', 'fwd_seg_size_min', 'fwd_act_data_pkts', 'flow_iat_mean', 'flow_iat_max', 'flow_iat_min', 'flow_iat_std', 'fwd_iat_tot', 'fwd_iat_max', 'fwd_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_tot', 'bwd_iat_max', 'bwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'fin_flag_cnt', 'syn_flag_cnt', 'down_up_ratio', 'pkt_size_avg', 'init_fwd_win_byts', 'init_bwd_win_byts', 'active_max', 'active_min', 'active_mean', 'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std', 'fwd_byts_b_avg', 'fwd_pkts_b_avg', 'bwd_byts_b_avg', 'bwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_blk_rate_avg', 'fwd_seg_size_avg', 'bwd_seg_size_avg', 'subflow_fwd_pkts', 'subflow_bwd_pkts', 'subflow_fwd_byts', 'subflow_bwd_byts']
SDN_COLUMNS = ['dst_port', 'protocol', 'flow_duration', 'tot_fwd_pkts', 'tot_bwd_pkts',
       'totlen_fwd_pkts', 'totlen_bwd_pkts', 'fwd_pkt_len_max',
       'fwd_pkt_len_min', 'fwd_pkt_len_mean', 'fwd_pkt_len_std',
       'bwd_pkt_len_max', 'bwd_pkt_len_min', 'bwd_pkt_len_mean',
       'bwd_pkt_len_std', 'flow_byts_s', 'flow_pkts_s', 'flow_iat_mean',
       'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_tot',
       'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min',
       'bwd_iat_tot', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max',
       'bwd_iat_min', 'bwd_psh_flags', 'bwd_urg_flags', 'fwd_header_len',
       'bwd_header_len', 'fwd_pkts_s', 'bwd_pkts_s', 'pkt_len_min',
       'pkt_len_max', 'pkt_len_mean', 'pkt_len_std', 'pkt_len_var',
       'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt',
       'ack_flag_cnt', 'urg_flag_cnt', 'down_up_ratio', 'pkt_size_avg',
       'fwd_seg_size_avg', 'bwd_seg_size_avg', 'fwd_byts_b_avg',
       'fwd_pkts_b_avg', 'fwd_blk_rate_avg', 'bwd_byts_b_avg',
       'bwd_pkts_b_avg', 'bwd_blk_rate_avg', 'subflow_fwd_pkts',
       'subflow_fwd_byts', 'subflow_bwd_pkts', 'subflow_bwd_byts',
       'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_act_data_pkts',
       'fwd_seg_size_min', 'active_mean', 'active_std', 'active_max',
       'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min']

def readlines_then_tail(fin):
    "Iterate through lines and then tail for further lines."
    while True:
        line = fin.readline()
        if line:
            yield line
        else:
            tail(fin)


def tail(fin):
    "Listen for new lines added to file."
    while True:
        where = fin.tell()
        line = fin.readline()
        if not line:
            time.sleep(SLEEP_INTERVAL)
            fin.seek(where)
        else:
            yield line


def listen():
    with open(args.data_path, 'r') as fin:
        encoder = load_model(args.encoder_path)
        count = 0
        features = ""
        for line in readlines_then_tail(fin):
            count += 1
            #print(line.strip())
            features = features + line.strip() + "\n"
            if count == args.batch_size:
                packet_df = read_packet(features)
                data_loader = process_packets(packet_df)
                detect(encoder, data_loader)
                count = 0
                features = ""



def read_packet(features):
  df = pd.read_csv(StringIO(features), sep=',', names=ORIGINAL_COLUMNS, header=None)
  df = df[INSECLAB_PREPROCESSED_COLUMNS]
  return df

def min_max_scaler(df):
	max_values = [65535, 17, 148711219.78759766, 5512, 14822, 30674315.0, 33731973.0, 64326.0, 8679.0, 21413.0, 33767.25643, 65394.0, 6525.0, 16059.75, 32119.5, 1077389044.8695652, 8388608.0, 119000000.0, 57700000.0, 120000000.0, 119000000.0, 148711219.78759766, 69400000.0, 78100000.0, 110000000.0, 69400000.0, 140871867.65670776, 81900000.0, 57700000.0, 120000000.0, 81900000.0, 1, 1, 110240, 296440, 4194304.0, 4194304.0, 8679, 65394, 18354.0, 31345.422389999996, 983000000.0, 1, 1, 1, 1, 1, 1, 313.6666666666667, 21413.0, 21413.0, 16059.75, 6372734.0, 2198.0, 498922447.2380952, 7275755.0, 2721.0, 804583212.137931, 5512, 30674315, 14822, 33731973, 65535, 65535, 5512, 20, 67000000.0, 30475087.28504181, 67000000.0, 67000000.0, 120000000.0, 38100000.0, 120000000.0, 120000000.0, 1]
	min_values = [0, 0, -151.0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -208053.6913, -13698.630140000001, -151.0, 0.0, -151.0, -7385.492324829102, 0.0, 0.0, 0.0, 0.0, -1203.5369873046875, -151.0, -151.0, 0.0, -151.0, -519.0372467041017, 0, 0, 0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, -1, -1, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
	for i in range(len(df.columns)):
		max = max_values[i]
		min = min_values[i]
		if (max != min):
			df[df.columns[i]] = df[df.columns[i]].map(lambda x : (x - min)/(max - min))
	return df

def process_packets(packet_df): # without labels
	#packet_df = min_max_scaler(packet_df)
	packet_df = packet_df.astype(np.float32)
	#packet_df = torch.tensor(packet_df.values)
	scaler = preprocessing.MinMaxScaler()
	packet_df = scaler.fit_transform(packet_df)
	packet_df = torch.from_numpy(packet_df)
	return packet_df

def detect(Q, data_loader):
	Q.eval()
	#X = Variable(data_loader)
	# Reconstruction phase
	output = Q(data_loader)[1]
	pred = output.data.max(1)[1]
	print(pred)
    # for i in pred:
    #     if i:
    #         print("Attack!!!")
    #     else:
    #         print("Normal!!!")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(args.n_features, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
        )

        self.lin_D = nn.Linear(1000, args.latent_dim)
        self.lin_D_cat = nn.Sequential(nn.Linear(1000, args.n_classes),nn.Softmax(1))


    def forward(self, img):
        x = self.model(img)

        z_gauss = self.lin_D(x)
        z_cat = self.lin_D_cat(x)

        return z_gauss, z_cat

def load_model(path):
  model = torch.load(path, map_location=torch.device('cpu'))
  return model

listen()


