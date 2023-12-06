import glob
import pandas as pd
import codecs
import json


class Train_Val_df:
    # def __init__(self):
    #     self.generate_df_from_json()

    def processing_dataset(self , text):
            """ Use a phone map and convert phoneme sequence to an integer sequence """
            phone_list = text.split('_2')
            phone_list.pop()
            int_sequence = []
            # print(phone_list)

            sentence = ""
            for phone_per_word in phone_list:
                phone_per_word = phone_per_word.lstrip()
                phone_per_word = phone_per_word.replace("_1", "")
                phone_per_word = phone_per_word.replace(" ", "")
                sentence += phone_per_word
                sentence += "@"
            return sentence
    
    @staticmethod
    def generate_df_from_dir(self):
        flac_path_train = glob.glob('/home/asif/Datasets_n_models_checkpoints/dataset_grapheme_cleaned/train/*.flac')
        flac_path_val = glob.glob("/home/asif/Datasets_n_models_checkpoints/dataset_grapheme_cleaned/valid/*.flac")
        # txt_path_train = glob.glob("/content/dataset/train/*.txt")
        # txt_path_val = glob.glob("/content/dataset/valid/*.txt")
        
        txt_list_train = []
        txt_list_val = []
        train_df = pd.DataFrame()
        for i in flac_path_train:
            # print(i)
            # print("///")
            i = i.replace("flac","txt")
            # print(i)
            with codecs.open(i, 'r', 'utf-8') as fid:
                for line in fid.readlines():
                    # processed_line = self.processing_dataset(line)
                    # txt_list_train.append(processed_line)
                    txt_list_train.append(line)
        

        for i in flac_path_val:
            i = i.replace("flac","txt")
            with codecs.open(i, 'r', 'utf-8') as fid:
                for line in fid.readlines():
                    # processed_line = self.processing_dataset(line)
                    # txt_list_val.append(processed_line)
                    txt_list_val.append(line)


        # flac_path_train.sort()

        data_train = {
            'audio': flac_path_train,
            'sentence':txt_list_train,
        }

        data_val = {
            'audio': flac_path_val,
            'sentence':txt_list_val,
        }
        train_df = pd.DataFrame(data_train)
        val_df = pd.DataFrame(data_val)
        return train_df , val_df


    @staticmethod
    def generate_df_from_json(json_data, train_ratio=0.9):
        flac_path = []
        txt_list = []

        for key, value in json_data.items():
            audio_path = value['audio']

            audio_path = "/home/asif/DATASET/"+audio_path.split("/")[-1]
            text = value['text']

            flac_path.append(audio_path)
            txt_list.append(text)

        data = {
            'audio': flac_path,
            'sentence': txt_list,
        }

        df = pd.DataFrame(data)

        # Shuffle the DataFrame
        df = df.sample(frac=1).reset_index(drop=True)

        # Calculate the split index
        split_index = int(train_ratio * len(df))

        # Split the DataFrame into train and validation DataFrames
        train_df = df[:split_index]
        val_df = df[split_index:]

        return train_df, val_df


if __name__ == "__main__":
    # Read the JSON file
    with open('/home/asif/merged_dict.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Call the generate_df_from_json() method on the Train_Val_df class directly
    train_df, val_df = Train_Val_df.generate_df_from_json(json_data)

    print(train_df)
    print(val_df)






"""
Some good code for avobe class
##############################
import glob
import pandas as pd
import codecs
import json

class Train_Val_df:
    def __init__(self, json_data=None, train_ratio=0.8):
        if json_data is not None:
            self.train_df, self.val_df = self.generate_df_from_json(json_data, train_ratio)
        else:
            # Call other functions here based on conditions or arguments

    @staticmethod
    def generate_df_from_json(json_data, train_ratio=0.8):
        # Function implementation remains the same

if __name__ == "__main__":
    # Read the JSON file
    with open('/home/asif/validated_jsons/csv2_validated_dict.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Instantiate the Train_Val_df class with the json_data
    tran_val_df = Train_Val_df(json_data=json_data)

    # Access the train_df and val_df attributes
    train_df = tran_val_df.train_df
    val_df = tran_val_df.val_df

    print(train_df)
    print(val_df)


################################
import glob
import pandas as pd
import codecs
import json

class Train_Val_df:
    def __init__(self, mode="from_json"):
        self.mode = mode

    def generate_df(self):
        if self.mode == "from_json":
            return self.generate_df_from_json
        else:
            # Add other function calls here based on the 'mode' value

    @staticmethod
    def generate_df_from_json(json_data, train_ratio=0.8):
        # Function implementation remains the same

if __name__ == "__main__":
    # Read the JSON file
    with open('/home/asif/validated_jsons/csv2_validated_dict.json', 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Instantiate the Train_Val_df class with the desired mode
    tran_val_df = Train_Val_df(mode="from_json")

    # Call the generate_df method which will return the appropriate function
    generate_df_func = tran_val_df.generate_df()

    # Call the returned function with the necessary arguments
    train_df, val_df = generate_df_func(json_data)

    print(train_df)
    print(val_df)
"""

