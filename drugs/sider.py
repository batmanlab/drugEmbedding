#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn import manifold
from sklearn.metrics.pairwise import cosine_similarity
import holoviews as hv


def make_freq_df(path=Path('/pghbio/dbmi/batmanlab/bpollack/drugEmbedding/sider')):
    # grab the frequency set
    df_freq = pd.read_csv(Path(path, 'meddra_freq.tsv.gz'), sep='\t', header=None,
                          names=['STICH_flat', 'STICH_stereo', 'UMLS_label',
                                 'from_placebo', 'freq_desc', 'freq_lower',
                                 'freq_upper', 'MedDRA_concept_type',
                                 'UMLS_MedDRA', 'side_effect'])
    # make a new column to account for items without quantiative frequency
    df_freq['freq_ave'] = (df_freq.freq_lower+df_freq.freq_upper)*0.5

    # grab the drug names
    df_names = pd.read_csv(Path(path, 'drug_names.tsv'), sep='\t', header=None,
                           names=['STICH_flat', 'name'])
    # merge with frequency
    df_freq = df_freq.merge(df_names, how='outer', on='STICH_flat').dropna(thresh=4)

    # grab the atc names
    df_atc = pd.read_csv(Path(path, 'drug_atc.tsv'), sep='\t', header=None,
                         names=['STICH_flat', 'atc'])
    # merge with frequency
    df_freq = df_freq.merge(df_atc, how='outer',
                            on='STICH_flat').dropna(thresh=4).drop_duplicates(['side_effect',
                                                                               'name'])

    return df_freq


def make_sider_vectors(df_freq):
    df_As = df_freq[df_freq.atc.str.startswith('A', na=False)]
    s_As = df_As.groupby('side_effect')['freq_ave'].sum()/df_As.name.nunique()
    s_As.name = 'Alimentary'

    df_Bs = df_freq[df_freq.atc.str.startswith('B', na=False)]
    s_Bs = df_Bs.groupby('side_effect')['freq_ave'].sum()/df_Bs.name.nunique()
    s_Bs.name = 'Blood'

    df_Cs = df_freq[df_freq.atc.str.startswith('C', na=False)]
    s_Cs = df_Cs.groupby('side_effect')['freq_ave'].sum()/df_Cs.name.nunique()
    s_Cs.name = 'Cardiovascular'

    df_Ds = df_freq[df_freq.atc.str.startswith('D', na=False)]
    s_Ds = df_Ds.groupby('side_effect')['freq_ave'].sum()/df_Ds.name.nunique()
    s_Ds.name = 'Dermatologicals'

    df_Gs = df_freq[df_freq.atc.str.startswith('G', na=False)]
    s_Gs = df_Gs.groupby('side_effect')['freq_ave'].sum()/df_Gs.name.nunique()
    s_Gs.name = 'Genitourinary'

    df_Hs = df_freq[df_freq.atc.str.startswith('H', na=False)]
    s_Hs = df_Hs.groupby('side_effect')['freq_ave'].sum()/df_Hs.name.nunique()
    s_Hs.name = 'Hormonal'

    df_Js = df_freq[df_freq.atc.str.startswith('J', na=False)]
    s_Js = df_Js.groupby('side_effect')['freq_ave'].sum()/df_Js.name.nunique()
    s_Js.name = 'J_Antiinfectives'

    df_Ls = df_freq[df_freq.atc.str.startswith('L', na=False)]
    s_Ls = df_Ls.groupby('side_effect')['freq_ave'].sum()/df_Ls.name.nunique()
    s_Ls.name = 'L_Antineoplastic'

    df_Ms = df_freq[df_freq.atc.str.startswith('M', na=False)]
    s_Ms = df_Ms.groupby('side_effect')['freq_ave'].sum()/df_Ms.name.nunique()
    s_Ms.name = 'Musculoskeletal'

    df_Ns = df_freq[df_freq.atc.str.startswith('N', na=False)]
    s_Ns = df_Ns.groupby('side_effect')['freq_ave'].sum()/df_Ns.name.nunique()
    s_Ns.name = 'Nervous'

    df_Ps = df_freq[df_freq.atc.str.startswith('P', na=False)]
    s_Ps = df_Ps.groupby('side_effect')['freq_ave'].sum()/df_Ps.name.nunique()
    s_Ps.name = 'P_Antiparasitic'

    df_Rs = df_freq[df_freq.atc.str.startswith('R', na=False)]
    s_Rs = df_Rs.groupby('side_effect')['freq_ave'].sum()/df_Rs.name.nunique()
    s_Rs.name = 'Respiratory'

    df_Ss = df_freq[df_freq.atc.str.startswith('S', na=False)]
    s_Ss = df_Ss.groupby('side_effect')['freq_ave'].sum()/df_Ss.name.nunique()
    s_Ss.name = 'Sensory'

    df_Vs = df_freq[df_freq.atc.str.startswith('V', na=False)]
    s_Vs = df_Vs.groupby('side_effect')['freq_ave'].sum()/df_Vs.name.nunique()
    s_Vs.name = 'Various'

    return pd.DataFrame([s_As, s_Bs, s_Cs, s_Ds, s_Gs, s_Hs, s_Js, s_Ls, s_Ms,
                         s_Ns, s_Ps, s_Rs, s_Ss, s_Vs])


def add_drugs_to_vec(df_freq, df_vec):
    # very slow first iteration
    # drug_vecs = []
    df_freq = df_freq.dropna(axis=0, subset=['atc', 'name']).sort_values('atc')
    for name in tqdm_notebook(df_freq.name.unique()):
        drug_vec = df_freq.query(f'name=="{name}"')
        atc = drug_vec.atc.iloc[0][0]
        drug_vec = drug_vec[['side_effect', 'freq_ave']]
        drug_vec = drug_vec.set_index('side_effect').rename(columns={'freq_ave': f'{atc}_{name}'}).T
        df_vec = df_vec.append(drug_vec)
    return df_vec


def tsne_plot(df_vec, group_list=(0, 1, 2), n_effects=100, perplexity=50):
    group_list = list(group_list)
    df_groups = df_vec.iloc[group_list]
    df_drugs = df_vec.iloc[14:]
    A = df_drugs.index.str.startswith('A_')
    B = df_drugs.index.str.startswith('B_')
    C = df_drugs.index.str.startswith('C_')
    D = df_drugs.index.str.startswith('D_')
    G = df_drugs.index.str.startswith('G_')
    H = df_drugs.index.str.startswith('H_')
    J = df_drugs.index.str.startswith('J_')
    L = df_drugs.index.str.startswith('L_')
    M = df_drugs.index.str.startswith('M_')
    N = df_drugs.index.str.startswith('N_')
    P = df_drugs.index.str.startswith('P_')
    R = df_drugs.index.str.startswith('R_')
    S = df_drugs.index.str.startswith('S_')
    V = df_drugs.index.str.startswith('V_')
    cat_index_list = np.asarray([A, B, C, D, G, H, J, L, M, N, P, R, S, V])
    cat_index_subset = cat_index_list[group_list].sum(axis=0)
    cat_index_subset = cat_index_subset.astype(bool)

    df_drugs_skim = df_drugs[cat_index_subset]
    top_index = df_groups.sum().sort_values(ascending=False)[:n_effects].index

    cs = cosine_similarity(df_drugs_skim[top_index], df_groups[top_index])
    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=perplexity)
    # Y = tsne.fit_transform(df_drugs_skim[top_index])
    Y = tsne.fit_transform(cs)

    df_labels = pd.DataFrame({'X': Y[:, 0], 'Y': Y[:, 1], 'Cat': 0}, index=df_drugs_skim.index)
    A = df_labels.index.str.startswith('A_')
    B = df_labels.index.str.startswith('B_')
    C = df_labels.index.str.startswith('C_')
    D = df_labels.index.str.startswith('D_')
    G = df_labels.index.str.startswith('G_')
    H = df_labels.index.str.startswith('H_')
    J = df_labels.index.str.startswith('J_')
    L = df_labels.index.str.startswith('L_')
    M = df_labels.index.str.startswith('M_')
    N = df_labels.index.str.startswith('N_')
    P = df_labels.index.str.startswith('P_')
    R = df_labels.index.str.startswith('R_')
    S = df_labels.index.str.startswith('S_')
    V = df_labels.index.str.startswith('V_')
    if 0 in group_list:
        df_labels.loc[A, 'Cat'] = 'Alimentary'
    if 1 in group_list:
        df_labels.loc[B, 'Cat'] = 'Blood'
    if 2 in group_list:
        df_labels.loc[C, 'Cat'] = 'Cardiovascular'
    if 3 in group_list:
        df_labels.loc[D, 'Cat'] = 'Dermatologicals'
    if 4 in group_list:
        df_labels.loc[G, 'Cat'] = 'Genitourinary'
    if 5 in group_list:
        df_labels.loc[H, 'Cat'] = 'Hormonal'
    if 6 in group_list:
        df_labels.loc[J, 'Cat'] = 'J_Antiinfectives'
    if 7 in group_list:
        df_labels.loc[L, 'Cat'] = 'L_Antineoplastic'
    if 8 in group_list:
        df_labels.loc[M, 'Cat'] = 'Musculoskeletal'
    if 9 in group_list:
        df_labels.loc[N, 'Cat'] = 'Nervous'
    if 10 in group_list:
        df_labels.loc[P, 'Cat'] = 'P_Antiparasitic'
    if 11 in group_list:
        df_labels.loc[R, 'Cat'] = 'Respiratory'
    if 12 in group_list:
        df_labels.loc[S, 'Cat'] = 'Sensory'
    if 13 in group_list:
        df_labels.loc[V, 'Cat'] = 'Various'

    df_labels_skim = df_labels.query('Cat!=0')
    df_labels_skim = df_labels_skim.reset_index().rename(columns={'index': 'name'})
    embed = hv.Points(df_labels_skim, kdims=['X', 'Y'], vdims=['Cat', 'name'],
                      label='Sider Embedding')
    return embed.opts(width=1000, height=800, color_index='Cat', cmap='colorblind', size=10,
                      tools=['hover'], legend_position='top_left', show_legend=True)
