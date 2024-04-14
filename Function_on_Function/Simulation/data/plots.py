import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



beta1 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/beta1/beta1.csv', header = None)
surface11 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B1_G1/true_surface/true_surface.csv', header = None)
surface12 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B1_G2/true_surface/true_surface.csv', header = None)
surface131 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B1_G3/true_surface1/true_surface1.csv', header = None)
surface132 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B1_G3/true_surface2/true_surface2.csv', header = None)

beta2 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/beta2/beta2.csv', header = None)
surface21 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B2_G1/true_surface/true_surface.csv', header = None)
surface22 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B2_G2/true_surface/true_surface.csv', header = None)
surface231 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B2_G3/true_surface1/true_surface1.csv', header = None)
surface232 = pd.read_csv('Datasets/Function_on_Function/Simulation/data/B2_G3/true_surface2/true_surface2.csv', header = None)




x = y = np.linspace(-2, 4, 100)
X, Y = np.meshgrid(x, y)

def plot3d(axis, beta, surface, beta_no, g_no, double = False, shade = False):

    shade = True
    if beta_no == 1:
        beta_color = 'crimson'
        surface1_color = 'denim'
        surface2_color = 'forest'
        x0, y0 = -0.25, 0.0
    elif beta_no == 2:
        beta_color = 'crimson'
        surface1_color = 'denim'
        surface2_color = 'forest'
        x0, y0 = 0.65, 0.0

    
    if g_no == 1:
        label_surface1 = r'$\beta_{' + str(beta_no) + r'}(s,t) \tilde{X}(t)$'
        bbox_to_anchor = (x0, y0, 1, 1)
    elif g_no == 2:
        label_surface1 = r'$\sqrt{1 + \left|\beta_{' + str(beta_no) + r'}(s,t) \tilde{X}(t)\right|}$'
        bbox_to_anchor = (x0, y0, 1, 1)
    elif g_no == 3:
        label_surface1 = r'$\beta_{' + str(beta_no) + r'}(s,t) \tilde{X}(t) \text{ for } \bar{X}(t) < 2.5$'
        label_surface2 = r'$3 - \sqrt{\left|\beta_{' + str(beta_no) + r'}(s,t) \tilde{X}(t) + 2 \right|} \text{ for } \bar{X}(t) \geq 2.5$'
        bbox_to_anchor = (x0, y0 + 0.05, 1, 1)

    
    label_beta = r'$\beta_{' + str(beta_no) + r'}(s,t)$'

    if double == False:
        surface[surface > 6] = 0
        axis.plot_surface(X, Y, beta, label = label_beta, color = f'xkcd:{beta_color}', shade=shade)
        axis.plot_surface(X, Y, surface, label = label_surface1, color = f'xkcd:{surface1_color}', shade=shade)
        proxy1 = mpatches.Patch(color=f'xkcd:{beta_color}')
        proxy2 = mpatches.Patch(color=f'xkcd:{surface1_color}')
        axis.legend([proxy1, proxy2], [label_beta, label_surface1], loc='upper left', fontsize=6, frameon=0, bbox_to_anchor=bbox_to_anchor)
    else:
        surface[0][surface[0] > 6] = 0
        surface[1][surface[1] > 6] = 0
        axis.plot_surface(X, Y, beta, label = label_beta, color = f'xkcd:{beta_color}', shade=shade)
        axis.plot_surface(X, Y, surface[0], label = label_surface1, color = f'xkcd:{surface1_color}', shade=shade)
        axis.plot_surface(X, Y, surface[1], label = label_surface2, color = f'xkcd:{surface2_color}', shade=shade)
        proxy1 = mpatches.Patch(color=f'xkcd:{beta_color}')
        proxy2 = mpatches.Patch(color=f'xkcd:{surface1_color}')
        proxy3 = mpatches.Patch(color=f'xkcd:{surface2_color}')
        axis.legend([proxy1, proxy2, proxy3], [label_beta, label_surface1, label_surface2], loc='upper left', fontsize=6, frameon=0, bbox_to_anchor=bbox_to_anchor)
    
    #axis.legend(loc = 'upper left', fontsize = 6, frameon = 0, bbox_to_anchor = bbox_to_anchor)        
    axis.set_xlabel('s')
    axis.set_ylabel('t')
    axis.set_xlim(-2, 4)
    axis.set_ylim(-2, 4)
    axis.set_zlim(-2, 6)
    return axis


fig = plt.figure(figsize=(9, 9))
fig.subplots_adjust(wspace=0, hspace=0)
ax1 = plt.subplot(3,2,1, projection='3d')
ax2 = plt.subplot(3,2,3, projection='3d')
ax3 = plt.subplot(3,2,5, projection='3d')
ax1.view_init(elev=10, azim=-135, roll=0)
ax2.view_init(elev=10, azim=-135, roll=0)
ax3.view_init(elev=10, azim=-135, roll=0)
ax4 = plt.subplot(3,2,2, projection='3d')
ax5 = plt.subplot(3,2,4, projection='3d')
ax6 = plt.subplot(3,2,6, projection='3d')
ax4.view_init(elev=10, azim=195, roll=0)
ax5.view_init(elev=10, azim=195, roll=0)
ax6.view_init(elev=10, azim=195, roll=0)
ax1 = plot3d(ax1, beta1, surface11, 1, 1)
ax2 = plot3d(ax2, beta1, surface12, 1, 2)
ax3 = plot3d(ax3, beta1, [surface131, surface132], 1, 3, True)
ax4 = plot3d(ax4, beta2, surface21, 2, 1)
ax5 = plot3d(ax5, beta2, surface22, 2, 2)
ax6 = plot3d(ax6, beta2, [surface231, surface232], 2, 3, True)
fig.savefig('Datasets/Function_on_Function/Simulation/data/beta_weights_functional.png', dpi=600, transparent=True)