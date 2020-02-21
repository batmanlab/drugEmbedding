import numpy as np
from lorentz import lorentz_to_poincare
import matplotlib.pyplot as plt

def poincare_scatter_plot(DrugsLoader, model, epoch):

    # create color dictionary
    color_dict = {}
    color_dict['alimentary tract and metabolism drugs'] = 0
    color_dict['antiinfectives for systemic use'] = 1
    color_dict['antineoplastic and immunomodulating agents'] = 2
    color_dict['antiparasitic products, insecticides and repellents'] = 3
    color_dict['blood and blood forming organ drugs'] = 4
    color_dict['cardiovascular system drugs'] = 5
    color_dict['dermatologicals'] = 6
    color_dict['genito urinary system and sex hormones'] = 7
    color_dict['musculo-skeletal system drugs'] = 8
    color_dict['nervous system drugs'] = 9
    color_dict['respiratory system drugs'] = 10
    color_dict['sensory organ drugs'] = 11
    color_dict['systemic hormonal preparations, excl. sex hormones and insulins'] = 12
    color_dict['various drug classes in atc'] = 13

    # freeze model and plot
    for iteration, batch in enumerate(DrugsLoader):
        recon_loss, local_ranking_loss, mean_z = model(batch['drug_inputs'], batch['drug_len'], batch['drug_targets'],
                                                            batch['loc_ranking_inputs'], batch['loc_ranking_len'], nneg,
                                                            batch['loc_ranking_sp'])
        # save drug reps. for visualization
        if iteration == 0:
            drug_lor_reps = mean_z.detach().numpy()
            drug_poi_reps = lorentz_to_poincare(drug_lor_reps)
            color_lst = [color_dict[x] for x in batch['ATC_LVL1']]
        else:
            batch_poi_reps = lorentz_to_poincare(mean_z.detach().numpy())
            drug_poi_reps = np.concatenate((drug_poi_reps, batch_poi_reps))
            color_lst = color_lst + [color_dict[x] for x in batch['ATC_LVL1']]

    # freeze model and plot
    z1 = drug_poi_reps[:, 0]
    z2 = drug_poi_reps[:, 1]
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10))
    scatter = ax.scatter(z1, z2, c=color_lst, cmap='tab20')
    handles, labels = scatter.legend_elements(prop='colors')
    labels = list(color_dict.keys())
    legend = ax.legend(handles, labels, title='ATC 1st level: anatomical groups', title_fontsize=14,
              loc='upper center', bbox_to_anchor=(0.5, -0.2), borderaxespad=0, ncol=2, frameon=False)
    circle = plt.Circle((0, 0), 1, color='k', fill=False)
    ax.add_artist(circle)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Epoch ' + str(epoch))
    fig.savefig('./experiments/local_ranking_2D/' + 'Epoch_' + str(epoch), bbox_extra_artists=(legend,), bbox_inches='tight')
