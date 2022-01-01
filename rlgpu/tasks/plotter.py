from inspect import CO_ITERABLE_COROUTINE
from numpy import log
from typing import Sequence
from copy import copy

import argparse
from tensorboard.backend.event_processing import event_accumulator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
plt.figure(figsize=(6, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框

font1 = {'family' : 'Times New Roman',  
'weight' : 'normal',  
'size'   : 9,  
} 

font2 = {'family' : 'Times New Roman',  
'weight' : 'normal',  
'size'   : 14,  
}  

def tensorboard_smoothing(arr: Sequence, smooth: float = 0.9) -> Sequence:
    """tensorboard smoothing  底层算法实现

    :param arr: shape(N,). const.
    :param smooth: smoothing系数
    :return: new_x
    """
    arr = copy(arr)
    weight = smooth  # 权重 (动态规划)
    for i in range(1, len(arr)):
        arr[i] = (arr[i - 1] * weight + arr[i]) / (weight + 1)
        weight = (weight + 1) * smooth  # `* smooth` 是为了让下一元素 权重为1
    return arr


def plot(params):
  ''' beautify tf log
      Use better library (seaborn) to plot tf event file'''

  log_paths = []
  log_paths.append("/home/lohse/isaac_ws/src/isaac-gym/scripts/Isaac-drlgrasp/rlgpu/logs/baxter_cabinet_re-model")
  log_paths.append("/home/lohse/isaac_ws/src/isaac-gym/scripts/Isaac-drlgrasp/rlgpu/logs/baxter_cabinet")
  smooth_space = params['smooth']
  color_code = params['color']
  x_list = []
  y_list = []
  for log_path in log_paths:
    print(log_path)
    acc = ea.EventAccumulator(log_path)
    acc.Reload()

    # only support scalar now
    scalar_list = acc.Tags()['scalars']

    x_list_raw = []
    y_list_raw = []
    for tag in scalar_list:
      if tag == "Reward/Reward":
        x = [int(s.step) for s in acc.Scalars(tag)]
        y = [s.value for s in acc.Scalars(tag)]
        
        
        # smooth curve
        x_ = []
        y_ = []
        for i in range(0, len(x)):
          if len(x) > 5000:
            if i < len(x) - 1625:
              continue
          x_.append(x[i])
          y_.append(y[i])
        y_ = tensorboard_smoothing(y_, 0.6)    
        x_list.append(x_)
        y_list.append(y_)


  plt.figure(1)
  plt.subplot(111)

  plt.plot(x_list[0], y_list[0], color='deepskyblue', label='$with \, demonstration$', linewidth=0.8)  # 绘制，指定颜色、标签、线宽，标签采用latex格式
  hl=plt.legend(loc='upper left', prop=font1)

  plt.plot(x_list[1], y_list[1], color='magenta', label='$w/o \, demonstration$', linewidth=0.8)
  plt.legend(loc='upper left', prop=font1)                                # 绘制图例，指定图例位置
  plt.xlim(0, 1600)
  leg = plt.gca().get_legend()
  ltext = leg.get_texts()
  plt.xlabel("Episodes", fontsize=9, fontweight='normal')
  plt.ylabel("Reward", fontsize=9, fontweight='normal')

  plt.savefig('./filename.pdf', format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
  plt.show()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--smooth', default=1, type=float, help='window size for average smoothing')
  parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  plot(params)
