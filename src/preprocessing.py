import pandas as pd
import numpy as np
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
         [x_train, y_train],
         axis = 1
     )
    valid_set = pd.concat(
         [x_valid, y_valid],
         axis = 1
     )
    test_set = pd.concat(
        [x_test, y_test],
         axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set


def nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace -1 with NaN
    set_data.replace(
        -1, np.nan,
        inplace = True
    )

    # Return replaced set data
    return set_data

def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:
    # Create ohe object
    ohe_continent = OneHotEncoder(sparse = False)

    # Fit ohe
    ohe_continent.fit(np.array(data_tobe_fitted).reshape(-1, 1))

    # Save ohe object
    util.pickle_dump(
        ohe_continent,
        ohe_path
    )

    # Return trained ohe
    return ohe_continent

def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_path: str) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Load ohe stasiun
    ohe_continent = util.pickle_load(ohe_path)

    # Transform variable stasiun of set data, resulting array
    continent_features = ohe_continent.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    # Convert to dataframe
    continent_features = pd.DataFrame(continent_features.tolist(), columns = list(ohe_continent.categories_[0]))

    # Set index by original set data index
    continent_features.set_index(set_data.index, inplace = True)

    # Concatenate new features with original set data
    set_data = pd.concat([continent_features, set_data], axis = 1)

    # Drop continent column
    #set_data.drop(columns = "continent", inplace = True)

    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    # Return new feature engineered set data
    return set_data


def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    rus = RandomUnderSampler(random_state = 26)

    # Balancing set data
    x_rus, y_rus = rus.fit_resample(set_data.drop("continent", axis = 1), set_data.continent)

    # Concatenate balanced data
    set_data_rus = pd.concat([x_rus, y_rus], axis = 1)

    # Return balanced data
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 11)

    # Balancing set data
    x_ros, y_ros = ros.fit_resample(set_data.drop("continent", axis = 1), set_data.continent)

    # Concatenate balanced data
    set_data_ros = pd.concat([x_ros, y_ros], axis = 1)

    # Return balanced data
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    x_sm, y_sm = sm.fit_resample(set_data.drop("continent", axis = 1), set_data.continent)

    # Concatenate balanced data
    set_data_sm = pd.concat([x_sm, y_sm], axis = 1)

    # Return balanced data
    return set_data_sm



if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)


    # 3. Converting -1 to NaN
    train_set = nan_detector(train_set)
    valid_set = nan_detector(valid_set)
    test_set = nan_detector(test_set)

    

    # 8. Fit ohe with predefined continent data
    ohe_continent = ohe_fit(
        config_data["range_continent"],
        config_data["ohe_continent_path"]
    )

    # 9. Transform stasiun on train, valid, and test set
    train_set = ohe_transform(
        train_set,
        "continent",
        config_data["ohe_continent_path"]
    )

    valid_set = ohe_transform(
        valid_set,
        "continent",
        config_data["ohe_continent_path"]
    )

    test_set = ohe_transform(
        test_set,
        "continent",
        config_data["ohe_continent_path"]
    )

    # 10. Undersampling dataset
    train_set_rus = rus_fit_resample(train_set)

    # 11. Oversampling dataset
    train_set_ros = ros_fit_resample(train_set)

    # 12. SMOTE dataset
    train_set_sm = sm_fit_resample(train_set)

    

    # 19. Dumping dataset
    x_train = {
        "Undersampling" : train_set_rus.drop(columns = "continent"),
        "Oversampling" : train_set_ros.drop(columns = "continent"),
        "SMOTE" : train_set_sm.drop(columns = "continent")
    }

    y_train = {
        "Undersampling" : train_set_rus.continent,
        "Oversampling" : train_set_ros.continent,
        "SMOTE" : train_set_sm.continent
    }

    util.pickle_dump(
        x_train,
        "data/processed/x_train_feng.pkl"
    )
    util.pickle_dump(
        y_train,
        "data/processed/y_train_feng.pkl"
    )

    util.pickle_dump(
        valid_set.drop(columns = "continent"),
        "data/processed/x_valid_feng.pkl"
    )
    util.pickle_dump(
        valid_set.continent,
        "data/processed/y_valid_feng.pkl"
    )

    util.pickle_dump(
        test_set.drop(columns = "continent"),
        "data/processed/x_test_feng.pkl"
    )
    util.pickle_dump(
        test_set.continent,
        "data/processed/y_test_feng.pkl"
    )