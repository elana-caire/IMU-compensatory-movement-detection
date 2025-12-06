#!/usr/bin/env python3
import os
import argparse
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode_condition_labels(df):
    """
    Utility function to encoder labels for conditions
    """

    df.loc[:,'Label'] = -1

    mask = df['condition'] == 'natural'
    df.loc[mask, 'Label'] = 0
    mask = df['condition'] == 'elbow_brace'
    df.loc[mask, 'Label'] = 1
    mask = df['condition'] == 'elbow_wrist_brace'
    df.loc[mask, 'Label'] = 2
    
    return df


def return_feature_columns(df, sensors_to_consider, 
                           time_features:list, 
                           frequency_features:list, 
                           exclude_quat = False, 
                           exclude_acc=False, 
                           exclude_gyro = False, 
                           exclude_mag = False):
    import itertools
    feat_columns = []
    for sensor in sensors_to_consider:
        if time_features is not None:
            for time_feat in time_features:
                quaternion_columns = (
                    df.columns[df.columns.str.contains("Quat") & df.columns.str.contains(sensor) & df.columns.str.contains(time_feat)]
                    if not exclude_quat else []
                )
                acc_columns = (
                    df.columns[df.columns.str.contains("Acc") & df.columns.str.contains(sensor) & df.columns.str.contains(time_feat) ]
                    if not exclude_acc else []
                )
                gyr_columns = (
                    df.columns[df.columns.str.contains("Gyr") & df.columns.str.contains(sensor) & df.columns.str.contains(time_feat) ]
                    if not exclude_gyro else []
                )
                mag_columns = (
                    df.columns[df.columns.str.contains("Mag")& df.columns.str.contains(sensor) & df.columns.str.contains(time_feat)]
                    if not exclude_mag else []
                )
                feat_columns.append(list(quaternion_columns) + list(acc_columns) + list(gyr_columns) + list(mag_columns))
        if frequency_features is not None:
            for freq_feat in frequency_features:
                quaternion_columns = (
                    df.columns[df.columns.str.contains("Quat") & df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat)]
                    if not exclude_quat else []
                )
                acc_columns = (
                    df.columns[df.columns.str.contains("Acc") & df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat) ]
                    if not exclude_acc else []
                )
                gyr_columns = (
                    df.columns[df.columns.str.contains("Gyr") & df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat) ]
                    if not exclude_gyro else []
                )
                mag_columns = (
                    df.columns[df.columns.str.contains("Mag")& df.columns.str.contains(sensor) & df.columns.str.contains(freq_feat)]
                    if not exclude_mag else []
                )
                feat_columns.append(list(quaternion_columns) + list(acc_columns) + list(gyr_columns) + list(mag_columns))
        


    return list(itertools.chain.from_iterable(feat_columns))


def ensure_features_only(df, feature_cols):
    """
    Return X (numpy) and feature column list verifying presence.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    X = df[feature_cols].to_numpy(dtype=float)
    return X

