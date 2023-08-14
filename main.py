
from argparse import ArgumentParser
import logging
from datasets import load_dataset
import pandas as pd
import gzip
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from multiprocessing import Pool, cpu_count

def parse_args():
    parser = ArgumentParser(
        description = ""
    )
    parser.add_argument(
        '--log_level', type=str, help='Set log level', default = "INFO"
    )
    parser.add_argument(
        '--log_file', type=str, help='Set log file'
    )
    return parser.parse_args()

def setup_logger(log_level='INFO', log_file=None):
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a' if log_file else 'w',
    )

    logger = logging.getLogger('my_script_logger')

    return logger

def gzip_compression_distance(x1, x2):
    Cx1 = len(gzip.compress(x1.encode()))
    Cx2 = len(gzip.compress(x2.encode()))
    x1x2 = " ".join([x1, x2])
    Cx1x2 = len(gzip.compress(x1x2.encode()))
    ncd = (Cx1x2 - min(Cx1,Cx2)) / max(Cx1, Cx2)

    return ncd

def get_features(df, text_column, target_column, new_text, k, ignore_zero_distance=False):
    ncd_distances = {}
    classes = {}

    logging.info(f"Gathering features for the following text: {new_text}")
    
    all_texts = df[text_column].to_list()
    
    distances = [gzip_compression_distance(text, new_text) for text in all_texts]
    for index, distance in enumerate(distances):
        if ignore_zero_distance and distance == 0:
            continue
        ncd_distances[index] = distance
        classes[index] = df.iloc[index][target_column]

    top_values_set = set(sorted(ncd_distances.values())[:k])
    
    class_counts = {class_id: [0] * k for class_id in set(classes.values())}

    for key, distance in ncd_distances.items():
        class_id = classes[key]
        if distance in top_values_set:
            value_index = list(top_values_set).index(distance)
            class_counts[class_id][value_index] += 1

    top_features = []
    for value, counts in zip(top_values_set, zip(*class_counts.values())):
        for count in counts:
            top_features.append(count / value)

    logging.info(top_features)

    return top_features

def get_features_parallel(args):
    df, text_column, target_column, new_text, k, ignore_zero_distance = args
    return get_features(df, text_column, target_column, new_text, k, ignore_zero_distance)

def main():
    args = parse_args()
    setup_logger(args.log_level, args.log_file)
    
    train_set = load_dataset("yahoo_answers_topics", split="train")
    test_set = load_dataset("yahoo_answers_topics", split="test")
    
    train_df = pd.DataFrame(train_set).drop(columns="id")
    test_df = pd.DataFrame(test_set).drop(columns="id")

    columns = ["question_title", "question_content"]
    top_k = 10
    test_column = "topic"

    for col in columns:
        with Pool(cpu_count()) as pool:  # Use all available CPUs
            all_texts_train = train_df[col].tolist()
            logging.info("Gathering training data")
            args_list_train = [(train_df, col, test_column, text, top_k, True) for text in all_texts_train]
            X_list_train = pool.map(get_features_parallel, args_list_train)
            X_train = pd.DataFrame(X_list_train)

            all_texts_test = test_df[col].tolist()
            logging.info("Gathering testing data")
            args_list_test = [(train_df, col, test_column, text, top_k, False) for text in all_texts_test]
            X_list_test = pool.map(get_features_parallel, args_list_test)
            X_test = pd.DataFrame(X_list_test)
            
            model = XGBClassifier()
            model.fit(X_train, train_df[test_column])
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(test_df[test_column], y_pred)
            logging.info(accuracy)

if __name__ == "__main__":
    main()