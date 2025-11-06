import pandas as pd


def feature_selection(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """
    Remove irrelevant columns from the DataFrame.

    Args:
        df: DataFrame with the data
        copy: If True, create a copy of the DataFrame
        
    Returns:
        DataFrame with selected columns
    """
    if copy:
        df = df.copy()

    to_drop = [
        'TruckOwner',
        'AdjustmentsToCreditRating',
        'BlockedCalls',
        'IncomeGroup',
        'OptOutMailings',
        'InboundCalls',
        'ReceivedCalls',
        'AgeHH2',
        'ReferralsMadeBySubscriber',
        'HandsetPrice',
        'DroppedBlockedCalls',
        'CallForwardingCalls',
        'OwnsComputer',
        'OverageMinutes',
        'DroppedCalls',
        'PercChangeRevenues',
        'ActiveSubs',
        'RoamingCalls',
        'MaritalStatus',
        'PeakCallsInOut',
        'HandsetModels',
        'ThreewayCalls',
        'RVOwner',
        'MadeCallToRetentionTeam',
        'NonUSTravel',
        'AgeHH1',
        'OffPeakCallsInOut',
        'OwnsMotorcycle',
        'HandsetRefurbished',
        'RetentionOffersAccepted',
        'CustomerID',
    ]

    to_drop = df.columns.intersection(to_drop)

    if len(to_drop) > 0:
        df = df.drop(to_drop, axis=1)

    return df