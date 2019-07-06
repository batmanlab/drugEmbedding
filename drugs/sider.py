#!/usr/bin/env python

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


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
