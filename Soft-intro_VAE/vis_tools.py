import cv2
import numpy as np
#import pdb
import plotly
from plotly.tools import mpl_to_plotly
from matplotlib import pyplot as plt
plt.switch_backend('agg')

class visualizer(object):

	def __init__(self, port = 8000, scatter_size=[[-1, 1], [-1, 1]]):
		import visdom
		self.vis = visdom.Visdom(port=port)
		(self.x_min, self.x_max), (self.y_min, self.y_max) = scatter_size
		self.counter = 0

	def img_result(self, img_list, caption = 'view', win = 1):
		self.vis.images(img_list,nrow = len(img_list), win = win, opts={'caption':caption})
	
	def plot_img_255(self, img, caption='view', win = 1):
		self.vis.image(img, win = win, opts={'caption': caption})

	# Occupies window 0
	def plot_error(self, errors, win = 0, id_val = 1):
		if not hasattr(self, 'plot_data'):
			self.plot_data = [{'X': [], 'Y': [], 'legend': list(errors.keys())}]
		elif len(self.plot_data) != id_val:
			self.plot_data.append({'X': [], 'Y': [], 'legend': list(errors.keys())})
		id_val -= 1
		self.plot_data[id_val]['X'].append(self.counter)
		self.plot_data[id_val]['Y'].append([errors[k]
									for k in self.plot_data[id_val]['legend']])
		self.vis.line(
			X=np.stack([np.array(self.plot_data[id_val]['X'])]*len(self.plot_data[id_val]['legend']), 1),
			Y=np.array(self.plot_data[id_val]['Y']),
			opts={
				'legend': self.plot_data[id_val]['legend'],
				'xlabel': 'epoch',
				'ylabel': 'loss'}, win = win)
		self.counter += 1

	def plot_quiver_img(self, img, flow, win = 0, caption = 'view'):
		fig, ax = plt.subplots(1)
		ax.axis('off')
		ax.imshow(img.transpose(1,2,0))
		X, Y, U, V = flow_to_XYUV(flow)
		ax.quiver(X, Y, U, V, angles='xy', color='y')
		plotly_fig = mpl_to_plotly(fig)
		self.vis.plotlyplot(plotly_fig)
		plt.clf()




























