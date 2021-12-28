from inspect import CO_ITERABLE_COROUTINE
from numpy import log
 
import argparse
from tensorboard.backend.event_processing import event_accumulator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper")

def plot(params):
  ''' beautify tf log
      Use better library (seaborn) to plot tf event file'''

  log_path = params['logdir']
  smooth_space = params['smooth']
  color_code = params['color']
  print(log_path)
  acc = ea.EventAccumulator(log_path)
  acc.Reload()

  # only support scalar now
  scalar_list = acc.Tags()['scalars']

  x_list = []
  y_list = []
  x_list_raw = []
  y_list_raw = []
  for tag in scalar_list:
    x = [int(s.step) for s in acc.Scalars(tag)]
    y = [s.value for s in acc.Scalars(tag)]

    # smooth curve
    x_ = []
    y_ = []
    for i in range(0, len(x), smooth_space):
      x_.append(x[i])
      y_.append(sum(y[i:i+smooth_space]) / float(smooth_space))    
    x_.append(x[-1])
    y_.append(y[-1])
    x_list.append(x_)
    y_list.append(y_)

    # raw curve
    x_list_raw.append(x)
    y_list_raw.append(y)


  for i in range(len(x_list)):
    # plt.figure(i)
    sns.relplot(x=x_list[i], y=y_list[i], kind="line", ci="sd")
    # plt.subplot(111)
    # plt.title(scalar_list[i])  
    # plt.plot(x_list_raw[i], y_list_raw[i], color=colors.to_rgba(color_code, alpha=0.4))
    # plt.plot(x_list[i], y_list[i], color=color_code, linewidth=1.5)
  plt.show()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', default='./logdir', type=str, help='logdir to event file')
  parser.add_argument('--smooth', default=6, type=float, help='window size for average smoothing')
  parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  plot(params)
