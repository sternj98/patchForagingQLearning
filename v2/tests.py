import scipy.io as sio

data = sio.loadmat('data/ql80_20200317.mat')
prt_n0_rews = data['prt_n0_rewbarcodes']

prts = prt_n0_rews[:,0]
n0 = prt_n0_rews[:,1]
rews = prt_n0_rews[:,2:]

session = [{"rews":}   for i in len(n0)]
