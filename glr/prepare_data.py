import numpy as np
import pandas as pd
from scipy import sparse
import argparse
import os


def prepare_data(data_name, min_interactions_per_user, remove_nan_skills, train_split=0.8):
    """Preprocess ASSISTments dataset.

    Arguments:
        data_name: "assistments09", "assistments12", "assistments15" or "assistments17"
        min_interactions_per_user (int): minimum number of interactions per student
        remove_nan_skills (bool): if True, remove interactions with no skill tag
        train_split (float): proportion of data to use for training

    Outputs:
        df (pandas DataFrame): preprocessed ASSISTments dataset with user_id, item_id,
            timestamp, correct and unique skill features
        Q_mat (item-skill relationships sparse array): corresponding q-matrix
    """
    data_path = os.path.join("../data/preprocessed", data_name)
    df = pd.read_csv(os.path.join(data_path, "data.csv"), encoding="ISO-8859-1")

    # Only 2012 and 2017 versions have timestamps
    if data_name == "assist09-updated":
        df["item_id"] = df["skill_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assist15":
        df = df.rename(columns={"sequence_id": "skill_id"})
        df["item_id"] = df["skill_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "assist17":
        df = df.rename(columns={"startTime": "timestamp",
                                "studentId": "user_id",
                                "skill": "skill_id"})
        df["item_id"] = df["skill_id"]
        df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    elif data_name == "intro-prog":
        df = df.rename(columns={"assignment_id": "skill_id"})
        df["item_id"] = df["skill_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "statics":
        df["item_id"] = df["skill_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "synthetic-5-k5":
        df["item_id"] = df["skill_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)
    elif data_name == "synthetic-5-k2":
        df["item_id"] = df["skill_id"]
        df["timestamp"] = np.zeros(len(df), dtype=np.int64)

    # Remove continuous outcomes
    df = df[df["correct"].isin([0, 1])]
    df["correct"] = df["correct"].astype(np.int32)

    # Filter nan skills
    if remove_nan_skills:
        df = df[~df["skill_id"].isnull()]
    else:
        df.loc[df["skill_id"].isnull(), "skill_id"] = -1

    # Filter too short sequences
    df = df.groupby("user_id").filter(lambda x: len(x) >= min_interactions_per_user)

    df["user_id"] = np.unique(df["user_id"], return_inverse=True)[1]
    df["item_id"] = np.unique(df["item_id"], return_inverse=True)[1]
    df["skill_id"] = np.unique(df["skill_id"], return_inverse=True)[1]

    # Build Q-matrix
    Q_mat = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
    for item_id, skill_id in df[["item_id", "skill_id"]].values:
        Q_mat[item_id, skill_id] = 1

    # Remove row duplicates due to multiple skills for one item
    if data_name == "assistments09":
        df = df.drop_duplicates("order_id")
    elif data_name == "assistments17":
        df = df.drop_duplicates(["user_id", "timestamp"])

    # Get unique skill id from combination of all skill ids
    unique_skill_ids = np.unique(Q_mat, axis=0, return_inverse=True)[1]
    df["skill_id"] = unique_skill_ids[df["item_id"]]

    # Sort data temporally
    if data_name in ["assistments12", "assistments17"]:
        df.sort_values(by="timestamp", inplace=True)
    elif data_name == "assistments09":
        df.sort_values(by="order_id", inplace=True)
    elif data_name == "assistments15":
        df.sort_values(by="log_id", inplace=True)

    # Sort data by users, preserving temporal order for each user
    df = pd.concat([u_df for _, u_df in df.groupby("user_id")])

    df = df[["user_id", "item_id", "timestamp", "correct", "skill_id"]]
    df.reset_index(inplace=True, drop=True)

    # Text files for BKT implementation (https://github.com/robert-lindsey/WCRP/)
    # bkt_dataset = df[["user_id", "item_id", "correct"]]
    # bkt_skills = unique_skill_ids
    # bkt_split = np.random.randint(low=0, high=5, size=df["user_id"].nunique()).reshape(1, -1)

    # Train-test split
    # users = df["user_id"].unique()
    # np.random.shuffle(users)
    # split = int(train_split * len(users))
    # train_df = df[df["user_id"].isin(users[:split])]
    # test_df = df[df["user_id"].isin(users[split:])]

    # Save data
    sparse.save_npz(os.path.join(data_path, "q_mat.npz"), sparse.csr_matrix(Q_mat))
    # train_df.to_csv(os.path.join(data_path, "preprocessed_data_train.csv"), sep="\t", index=False)
    # test_df.to_csv(os.path.join(data_path, "preprocessed_data_test.csv"), sep="\t", index=False)
    df.to_csv(os.path.join(data_path, "glr-preprocessed.csv"), sep="\t", index=False)
    # np.savetxt(os.path.join(data_path, "bkt_dataset.txt"), bkt_dataset, fmt='%i')
    # np.savetxt(os.path.join(data_path, "bkt_expert_labels.txt"), bkt_skills, fmt='%i')
    # np.savetxt(os.path.join(data_path, "bkt_splits.txt"), bkt_split, fmt='%i')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare datasets.')
    parser.add_argument('--dataset', type=str, default='assistments09')
    parser.add_argument('--min_interactions', type=int, default=2)
    parser.add_argument('--remove_nan_skills', action='store_true')
    args = parser.parse_args()

    prepare_data(
        data_name=args.dataset,
        min_interactions_per_user=args.min_interactions,
        remove_nan_skills=args.remove_nan_skills)
