import pandas as pd

class DowryDataNormalizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def normalize_family_type(self):
        self.df['family type'] = self.df['family type'].str.strip().str.lower()
        self.df['family type'] = self.df['family type'].replace({
            'rich class': 'Rich Class',
            'higher class': 'Higher Class',
            'higher middle class': 'Higher Class',  # merged here
            'lower class': 'Lower Class',
            'lower middle class': 'Lower Class',    # merged here
            'proverty level': 'Poverty Level',
            'poverty level': 'Poverty Level'
        })

    def normalize_girls_job(self):
        self.df["girl's job"] = self.df["girl's job"].str.strip().str.lower()
        self.df["girl's job"] = self.df["girl's job"].replace({
            'businessman': 'Business/Entrepreneur',
            'entrepreneur': 'Business/Entrepreneur',
            'service holder': 'Service Holder',
            'low service holder': 'Service Holder',
            'lower service holder': 'Service Holder',
            'garments worker': 'Worker',
            'lower worker': 'Worker',
            'day labour': 'Day Labour'
        })
        self.df["girl's job"] = self.df["girl's job"].replace({
            'business/entrepreneur': 'Business/Entrepreneur',
            'service holder': 'Service Holder',
            'worker': 'Worker',
            'day labour': 'Day Labour'
        })

    def normalize_boys_job(self):
        self.df["boy's job"] = self.df["boy's job"].str.strip().str.lower()
        self.df["boy's job"] = self.df["boy's job"].replace({
            'businessman': 'Business/Entrepreneur',
            'entrepreneur': 'Business/Entrepreneur',
            'service holder': 'Service Holder',
            'low service holder': 'Service Holder',
            'farmer': 'Farmer',
            'driver': 'Driver',
            'garments worker': 'Worker',
            'lower worker': 'Worker',
            'day labour': 'Day Labour'
        })
        self.df["boy's job"] = self.df["boy's job"].replace({
            'business/entrepreneur': 'Business/Entrepreneur',
            'service holder': 'Service Holder',
            'farmer': 'Farmer',
            'driver': 'Driver',
            'worker': 'Worker',
            'day labour': 'Day Labour'
        })

    def normalize_marry_condition(self):
        self.df["marry condition"] = self.df["marry condition"].str.strip().str.lower()
        self.df["marry condition"] = self.df["marry condition"].replace({
            'forced marriage': 'Forced Marriage',
            'irrelevant marriage': 'Irrelevant Marriage',
            'love marriage': 'Love Marriage',
            'arrange marriage': 'Arrange Marriage'
        })
        self.df["marry condition"] = self.df["marry condition"].replace({
            'forced marriage': 'Forced Marriage',
            'irrelevant marriage': 'Irrelevant Marriage',
            'love marriage': 'Love Marriage',
            'arrange marriage': 'Arrange Marriage'
        })

    def normalize_women_married(self):
        self.df["women married/unmarried"] = self.df["women married/unmarried"].str.strip().str.lower()
        self.df["women married/unmarried"] = self.df["women married/unmarried"].replace({
            'single': 'Single',
            'widow': 'Widow',
            'divorced': 'Divorced',
        })
        self.df["marry condition"] = self.df["marry condition"].replace({
            'single': 'Single',
            'widow': 'Widow',
            'divorced': 'Divorced',
        })

    def normalize_dowry(self):
        self.df["dowry"] = self.df["dowry"].str.strip().str.lower()
        self.df["dowry"] = self.df["dowry"].replace({
            'aurnaments': 'Aurnaments',
            'home': 'Home',
            'furniture': 'Furniture',
            'car': 'Car',
            'land': 'Land',
            'money': 'Money',
            'no property': 'No Property',
        })
        self.df["dowry"] = self.df["dowry"].replace({
            'aurnaments': 'Aurnaments',
            'home': 'Home',
            'furniture': 'Furniture',
            'car': 'Car',
            'land': 'Land',
            'money': 'Money',
            'no property': 'No Property',
        })

    def normalize_all(self):
        self.normalize_family_type()
        self.normalize_girls_job()
        self.normalize_boys_job()
        self.normalize_marry_condition()
        self.normalize_women_married()
        self.normalize_dowry()
        return self.df