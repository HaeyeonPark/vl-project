import os
import collections
import matplotlib.pyplot as plt

log_path = '/workspace/code/logs/exp5/exp5-hy.46609.txt'

#f_contents = open(log_path, 'r').read()
f = open(log_path, 'r')
lines = f.readlines()
lines = lines[77:]
loss_dict = collections.defaultdict(list)
loss_list = ['epoch','step','cmpm_loss', 'cmpc_loss','combine_loss' ,'cont_loss']

for i, line in enumerate(lines):
    if 'epoch' in line and 'step' in line:
        for loss_el in loss_list:
            if loss_el in line:
                s = line.find(loss_el)
                l = len(loss_el)
                v = line[s+l+1:].split(' ')[0]
                if v[-1]==',':
                    v = v[:-1]
                if 'loss' in loss_el:
                    loss_dict[loss_el].append(float(v))
                else:
                    loss_dict[loss_el].append(int(v))

steps_per_epoch = max(loss_dict['step'])
for i,epochs in enumerate(loss_dict['epoch']):
    loss_dict['total_step'].append(epochs * steps_per_epoch + loss_dict['step'][i])

color_dict = {'cmpm_loss':'brown', 'cmpc_loss':'darkgoldenrod', 'combine_loss':'seagreen', 'cont_loss':'slateblue'}
for k in loss_dict:
    if 'loss' in k:
        plt.plot(loss_dict['total_step'],loss_dict[k], color=color_dict[k], label=k)
plt.legend(loc='upper right')
plt.xlabel('step')
plt.ylabel('loss')
plt.savefig('tmp.png')
f.close()