This folder contains the re-trained model and tokenizer and its vocabulary files:

Here is the script used to save the model and tokenizer:

model_dir = './SavedModels/'
PATH = model_dir + "pytorch_distilbert_sent_uncased.pt"

# Save the model
torch.save(model, PATH)

#Save the tokenizer
tokenizer.save_pretrained(model_dir)

print('All files saved')

Here is the script used to retrieve the saved model and tokenizer:

model_dir = './SavedModels/'
PATH = model_dir + "pytorch_distilbert_sent_uncased.pt"

#Retrieve the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(model_dir)

#Retrieve the model
model = DistillBERTClass()
model = torch.load(PATH)
model.to(device)
model.eval()
