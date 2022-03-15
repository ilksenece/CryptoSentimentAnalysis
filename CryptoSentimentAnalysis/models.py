import logging
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import Dataset, DataLoader


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        tweet = str(self.data.Clean_Tweet[index])
        tweet = " ".join(tweet.split())
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.Numerical_Label[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len


def dataset_prep(df, train_size, rd_state, tokenizer, max_len):
    train_dataset = df.sample(frac=train_size, random_state=rd_state)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    logging.info("FULL Dataset: {}".format(df.shape))
    logging.info("TRAIN Dataset: {}".format(train_dataset.shape))
    logging.info("TEST Dataset: {}".format(test_dataset.shape))

    training_set = Triage(train_dataset, tokenizer, max_len)
    testing_set = Triage(test_dataset, tokenizer, max_len)
    all_set = Triage(df, tokenizer, max_len)

    return training_set, testing_set, all_set


def dataloader_prep(TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, training_set, testing_set, all_set):
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    full_loader = DataLoader(all_set, **train_params)
    return training_loader, testing_loader, full_loader


class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        # self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


# Function to calculate the accuracy of the model
def calculate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


# Defining the training function on the 80% of the dataset for tuning the distilbert model
def train(epoch, loss_function, optimizer, model, training_loader):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids']
        mask = data['mask']
        targets = data['targets']

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            logging.info(f"Training Loss per 5000 steps: {loss_step}")
            logging.info(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    logging.info('   ')
    logging.info(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    logging.info(f"Training Loss Epoch: {epoch_loss}")
    logging.info(f"Training Accuracy Epoch: {epoch_accu}")

    return


def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    logging.info(f"Accuracy: {acc}")
    logging.info(f"f1: {f1}")
    logging.info(f"precision: {precision}")
    logging.info(f"recall: {recall}")
    conf_mat = confusion_matrix(labels, preds)
    logging.info(f"Confusion matrix: {conf_mat}")
    cl_report = classification_report(labels, preds)
    logging.info(f"Classification report: {cl_report}")


def valid(model, testing_loader, loss_function):
    model.eval()
    empty_list = []
    predictions = np.array(empty_list)
    labels = np.array(empty_list)
    n_correct = 0
    n_wrong = 0
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids']
            mask = data['mask']
            targets = data['targets']
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accu(big_idx, targets)
            predictions = np.append(predictions, big_idx.to("cpu").numpy())
            labels = np.append(labels, targets.to("cpu").numpy())

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                logging.info(f"Validation Loss per 5000 steps: {loss_step}")
                logging.info(f"Validation Accuracy per 5000 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    logging.info(f"Validation Loss Epoch: {epoch_loss}")
    logging.info(f"Validation Accuracy Epoch: {epoch_accu}")
    compute_metrics(predictions, labels)

    return epoch_accu


#  model_name options
# 'distilbert-base-uncased-finetuned-sst-2-english'
# 'distilbert-base-uncased'
def retrain(df, model_name):
    # Defining some key variables that will be used later on in the training
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 4
    VALID_BATCH_SIZE = 2
    EPOCHS = 2
    LEARNING_RATE = 1e-05
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    training_set, testing_set, all_set = dataset_prep(df, 0.8, 200, tokenizer, MAX_LEN)
    training_loader, testing_loader, full_loader = dataloader_prep(TRAIN_BATCH_SIZE, VALID_BATCH_SIZE, training_set,
                                                                   testing_set, all_set)

    model = DistillBERTClass()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train(epoch, loss_function, optimizer, model, training_loader)

    acc = valid(model, testing_loader, loss_function)
    logging.info("Accuracy on test data = %0.2f%%" % acc)

    model_dir = './SavedModels/'
    # Specify a path
    path = model_dir + "pytorch_distilbert_sent_uncased.pt"

    # Save
    torch.save(model, path)
    tokenizer.save_pretrained(model_dir)

    logging.info('All files saved')
    return


def Tweet_Predict(new_tweet, label, tokenizer, model):
    # read from saved model and tokenizer
    #model = DistilBertTokenizer.from_pretrained("./SavedModels/")
    #tokenizer = DistilBertTokenizer.from_pretrained("./SavedModels/")

    MAX_LEN = 128
    VALID_BATCH_SIZE = 2

    print("Tweet", new_tweet)
    data = [[new_tweet, label]]

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Clean_Tweet', 'Numerical_Label'])
    df_set = Triage(df, tokenizer, MAX_LEN)

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    df_loader = DataLoader(df_set, **test_params)
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(df_loader, 0):
            ids = data['ids']  # .to(device, dtype=torch.long)
            mask = data['mask']  # .to(device, dtype=torch.long)
            targets = data['targets']  # .to(device, dtype=torch.long)
            outputs = model(ids, mask)
            big_val, big_idx = torch.max(outputs.data, dim=1)
    print("Result:", big_idx)
    return big_idx


def DF_Predict(data):
    # read from saved model and tokenizer
    model = DistilBertTokenizer.from_pretrained("./SavedModels/")
    tokenizer = DistilBertTokenizer.from_pretrained("./SavedModels/")

    MAX_LEN = 128
    VALID_BATCH_SIZE = 2

    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Clean_Tweet', 'Numerical_Label'])
    df_set = Triage(df, tokenizer, MAX_LEN)

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }

    df_loader = DataLoader(df_set, **test_params)
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(df_loader, 0):
            ids = data['ids']  # .to(device, dtype=torch.long)
            mask = data['mask']  # .to(device, dtype=torch.long)
            targets = data['targets']  # .to(device, dtype=torch.long)
            outputs = model(ids, mask)
            big_val, big_idx = torch.max(outputs.data, dim=1)
    return big_idx
