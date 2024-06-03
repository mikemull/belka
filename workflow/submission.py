
import pandas as pd
from pathlib import Path

from workflow.chemprop_predict import predict


def prepare_submission(test_df, checkpoints):
    """
    Produce a dataframe that has columns (id, binds) where binds here is the probability of binding
    Run separate model for each target and concatenate the results

    :param test_df:
    :param checkpoints:
    :return: dataframe with columns (id, binds)
    """
    targets = ('BRD4', 'sEH', 'HSA')
    pred_dfs = []

    for target in targets:
        print(target, checkpoints[target])
        target_df = test_df[test_df.protein_name == target].copy()
        pred_df = predict(checkpoints[target], target_df, smiles_column='canonical_smiles')
        pred_dfs.append(pred_df)

    return pd.concat(pred_dfs)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(prog='Belka workflow')
    parser.add_argument('test', type=Path, help='Path to the test dataset')
    parser.add_argument('checkpoints', type=Path, help='Path to the model checkpoints')
    args = parser.parse_args()

    test_df = pd.read_parquet(args.test)
    checkpoints = {f'{target}': args.checkpoints / f'{target}_model.ckpt' for target in ('BRD4', 'sEH', 'HSA')}

    submission_df = prepare_submission(test_df, checkpoints)
    submission_df.to_csv('submission.csv', index=False)
    print(submission_df.head())
    df_final = submission_df[['id', 'pred']].rename(columns={'pred': 'binds'})
    df_final.to_csv('submission.csv', index=False)
