def modify_vbdp_dataframe(df_vbdp):
    """Modify the dataframe with extended features for the vbdp model.
    Args:
        df_vbdp (pandas.DataFrame): The dataframe to modify.
    Returns:
        pandas.DataFrame: The modified dataframe.
    """
    # Determine the n most common diseases for each prognosis
    for n in range(3, 6):
        # Group the data by 'prognosis' column and count the occurrence of each disease
        disease_counts = df_vbdp.iloc[:, :64].groupby(df_vbdp["prognosis"]).sum()
        # Get the top n diseases for each prognosis
        top_diseases = disease_counts.apply(lambda x: x.nlargest(n).index.tolist(), axis=1)
        # Create new columns for each prognosis-disease combination
        for prognosis, diseases in top_diseases.items():
            for disease in diseases:
                column_name = f"{prognosis}_{disease}_{n}"
                df_vbdp[column_name] = 0
        # Iterate through the dataframe rows and update the new columns
        for index, row in df_vbdp.iterrows():
            prognosis = row["prognosis"]
            diseases = top_diseases[prognosis]
            for disease in diseases:
                column_name = f"{prognosis}_{disease}_{n}"
                if row[disease] == 1:
                    df_vbdp.at[index, column_name] = 1
    # Target should be the last column
    df_vbdp = df_vbdp[[c for c in df_vbdp if c not in ["prognosis"]] + ["prognosis"]]
    return df_vbdp
