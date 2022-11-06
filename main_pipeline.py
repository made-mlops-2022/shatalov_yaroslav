import os
import click
import logging

from modules.data_preparing import make_train_val_dataset
from modules.models import make_predictions_with_cls, fit_cls

def init_logging(log_lvl=logging.DEBUG, log_file_path=None):
    handlers = [logging.StreamHandler()]
    if log_file_path is not None:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_lvl)
        handlers.append(file_handler)
    logging.basicConfig(
        handlers=handlers,
        level=log_lvl,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )
    
@click.group()
@click.option('--log-file-path', default='logs\\model.log', help='File path for logs')
def main(log_file_path: str):
    init_logging(log_file_path=log_file_path)

@main.command()
@click.option('--df-file-path', default='data\\heart_cleveland_upload.csv', help='File path for data')
@click.option('--out-folder', default='data', help='File path for result datasets')
def make_datasets(df_file_path: str, out_folder: str):
    logging.info('==Creating datasets==')
    try:
        train_df, train_y, val_df, val_y = make_train_val_dataset(df_file_path)
    except Exception as ex:
        logging.error(f'An exception while creating dataset: {str(ex)}')
    
    logging.info('==Writing datasets==')
    try:
        train_df.to_csv(os.path.join(out_folder, 'train_df.csv'), index=False)
        train_y.to_csv(os.path.join(out_folder, 'train_y.csv'), index=False)
        val_df.to_csv(os.path.join(out_folder, 'val_df.csv'), index=False)
        val_y.to_csv(os.path.join(out_folder, 'val_y.csv'), index=False)
    except Exception as ex:
        logging.error(f'An exception while writing dataset: {str(ex)}')
    logging.info('==Datasets were created==')

@main.command()
@click.option('--train-df-file-path', default='data\\train_df.csv', help='File path for data')
@click.option('--train-y-file-path', default='data\\train_y.csv', help='File path for target')
@click.option('--cls-file-path', default='models\\cls.sav', help='File path for models')
def fit_model(train_df_file_path: str, train_y_file_path: str, cls_file_path: str):
    logging.info('==Start of fitting==')
    try:
        fit_cls(train_df_file_path, train_y_file_path, cls_file_path)
    except Exception as ex:
        logging.error(f'An exception while fitting model: {str(ex)}')
    logging.info('==Model fitted and saved==')


@main.command()
@click.option('--val-df-file-path', default='data\\val_df.csv', help='File path for data to make predictions')
@click.option('--cls-file-path', default='models\\cls.sav', help='File path for model')
@click.option('--res-predictions-file', default='data\\predictions.csv', help='File path for predictions')
def make_predictions(val_df_file_path: str, cls_file_path: str, res_predictions_file: str):
    logging.info('==Start of predicting process==')
    try:
        make_predictions_with_cls(val_df_file_path, cls_file_path, res_predictions_file)
    except Exception as ex:
        logging.error(f'An exception while fitting model: {str(ex)}')
    logging.info('==Predictions were made==')

if __name__ == '__main__':
    main()