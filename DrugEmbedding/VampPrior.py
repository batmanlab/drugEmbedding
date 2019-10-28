
def sample_fda_examplars():
    """
    temp function that random sample an examplar drug from ATC Level 2
    :return:
    """
    df_atc = pd.read_csv('./data/fda_drugs/atc_fda_unique.csv', index_col=0) # load ATC hierarchy inf.
    df_examplars = df_atc.groupby("ATC_LVL2").apply(lambda group_df: group_df.sample(1))

    with open('./data/fda_drugs/atc_examplars.txt', 'w') as fp:
        for idx, row in df_examplars.iterrows():
            fp.write(row['smile_x'] + ' ' + row['ATC_LVL5'] + ' \n')
    fp.close()

