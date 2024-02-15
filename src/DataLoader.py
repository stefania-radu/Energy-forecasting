import pandas as pd


class DataLoader:
    """
    Main class for data loading and data preprocessing
    """
    def __init__(self, file_path, split_ratio=0.1):
        self.file_path = file_path
        self.df = None
        self.history_data = None
        self.future_data = None
        self.parameters = None
        self.labels = None
        self.exog_variables = None
        self.split_ratio = split_ratio


    def load_data(self):
        """
        Load data from a .json file
        """
        self.df = pd.read_json(self.file_path)
        self._extract_exog_variables()
        self._extract_history_data()
        self._preprocess_history_data()
        self._extract_future_data()
        self._extract_parameters()
        self._extract_labels()
    
        return self


    def _extract_parameters(self):
        """
        Extract parameteres from the data
        """
        self.parameters = self.df['parameters']


    def _extract_labels(self):
        """
        Extract labels from the data
        """
        self.labels = self.df['labels']


    def _extract_history_data(self):
        """
        Generate a dataframe using the history data
        """
        history_times = self.df['history']['times']
        history_data = self.df['history']['data']['y']
        self.history_data = pd.DataFrame({
            'ds': pd.to_datetime(history_times).strftime('%Y-%m-%d %H:%M:%S'),
            'y': history_data
        })
        self.history_data["unique_id"] = self.df['labels']['asset_id']
        for variable in self.exog_variables:
            self.history_data[variable] = self.df['history']['data'][variable]


    def _extract_future_data(self):
        """
        Generate a dataframe using the future/test data
        """
        future_times = self.df['future']['times']
        self.future_data = pd.DataFrame({
           'ds': pd.to_datetime(future_times).strftime('%Y-%m-%d %H:%M:%S')
        })
        self.future_data["unique_id"] = self.df['labels']['asset_id']
        for variable in self.exog_variables:
            self.future_data[variable] = self.df['future']['data'][variable]
        

    def _preprocess_history_data(self):
        """
        Preprocess the history data by filling NaN values using backward fill and spliting the dataset into training and validation
        """
        self.history_data = self.history_data.bfill() # backward fill works best here

        # split data into train and validation
        split_index = int((1 - self.split_ratio) * len(self.history_data))
        
        self.history_data_train = self.history_data[:split_index]
        self.history_data_validation = self.history_data[split_index:]
        self.history_data_validation.loc[:, 'ds'] = pd.to_datetime(self.history_data_validation['ds'])


    def _extract_exog_variables(self):
        """
        Extract the exogenous variables
        """
        if 'data' in self.df['history']:
            self.exog_variables = list(self.df['history']['data'].keys())
            self.exog_variables.remove('y')  # Remove target variable from exogenous variables list





