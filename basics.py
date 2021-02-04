import spacy 
import torch
from flask import Flask, render_template, request
nlp = spacy.load('en_core_web_sm')


selected_keyphrases_list=[]
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

pattern1 = [{'POS':'ADJ', 'OP':'*'}, {'IS_PUNCT': True, 'OP':'*'}, {'POS':'NOUN', 'OP':'+'},{'IS_PUNCT': True, 'OP':'*'},{'POS':'ADP'},{'OP':'?'},{'POS':'ADJ'},{'IS_PUNCT': True, 'OP':'*'},{'POS':'NOUN'}]
#pat = [({'POS':'ADJ', 'OP':'*'}, {'IS_PUNCT': True, 'OP':'*'}, {'POS':'NOUN', 'OP':'+'},{'IS_PUNCT': True, 'OP':'*'},{'POS':'ADP'})?]
pattern2 = [{'POS':'ADJ'},{'IS_PUNCT': True, 'OP':'*'},{'POS':'NOUN'}]
pattern3 = [{'ENT_TYPE':'ORG', 'OP':'+'}]
pattern4 = [{'ENT_TYPE':'GPE', 'OP':'+'}]
matcher.add('Noun', None, pattern1, pattern2, pattern3, pattern4)

app = Flask(__name__)
#####################
@app.route('/')
def man():
    return render_template('home.html')

@app.route('/home', methods=['POST'])
def home():
    doc = request.files['text']
    doc.save('static/{}.txt')  
    f = open("static/{}.txt", "r")
    s = f.read()
    doc = nlp(s)

    found_matches = matcher(doc)
    phrases = set()
    for matcher_id, start, end in found_matches:
      if((start!=end) and (doc[start:end] not in nlp.Defaults.stop_words)):
        phrases.add(str(doc[start:end]))

    phrases=list(phrases)

    clean = list()
    punc = '''!()-[]{};:'"<>./?@#$%^&*_~\,'''
    for word in phrases:
      x = list(word)
      s=''
      for i in range(len(x)):
        if(x[i] in punc):
          continue
        else:
          s=s+x[i]
      clean.append(s)

    #clean = set(clean)
    clean = list(clean)
    clean.sort()
    print(len(clean))

    c=0
    new= list()
    for i in range(len(clean)):
      for j in range(len(clean)):
        if(i==j):
          continue
        if(clean[i] in clean[j]):
          c=1
          break
        c=0
      if(c==0):
        new.append(clean[i])
      c=0

    new = set(new)
    phrases = list(new)
    print(len(phrases))
    phrases


    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states = True,)
    model.eval()


    # In[12]:


    sentences= s.split('. ')
    #here we tokenize each sentence and convert to ids
    #finding maximum length of a sentence (for padding purposes, because bert requires this)
    max_len = 0
    # For every sentence...
    for sentence in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        sentence_id = tokenizer.encode(sentence, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(sentence_id))
    #adding 10 extra to max_len just in case
    max_len=max_len+10


    # In[13]:


    # Tokenize all of the sentences and map the tokens to their word IDs.
    # input ids is a 2d list, each element is a list(that represents a sentence which is tokenised and converted to the ids)
    #attention masks is also a 2d list each element is a list(that represents whether each element of the tokenised list is a padded element or not)
    input_ids = []
    attention_masks = []
    # For every phrase...
    for sentence in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the phrase.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the phrase to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sentence,                      # phrase to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            truncation = True,
                            max_length = 23 ,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        # Add the encoded phrase to the list.    
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])


    # In[14]:


    #this is where we call the model and pass the encoded tokens of each sentence
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    hidden_states=[]
    with torch.no_grad():
        for i in range(len(input_ids)):
          outputs = model(input_ids[i],attention_mask=attention_masks[i])
          hidden_states.append(outputs[2])

    print(len(hidden_states[0]))


    # In[15]:


    #here we calculate sentence and document embeddings
    # `hidden_states` has shape [no of sentences x 13 x 1 x max_len x 768]
    sentence_embeddings=[]
    # `token_vecs` is a tensor with shape [22 x 768]
    for i in range(len(hidden_states)):
      token_vecs = hidden_states[i][-2][0]#we take the embeddings from the second last layer as it has the best balance in terms of context
      # Calculate the average of all token vectors.
      sentence_embedding = torch.mean(token_vecs, dim=0)
      sentence_embeddings.append(sentence_embedding)

    sentence_embeddings = torch.stack(sentence_embeddings,dim=0)
    document_embedding=torch.mean(sentence_embeddings, dim=0)


    # In[16]:


    #finding maximum length of a phrase (for padding purposes, because bert requires this)
    max_len = 0

    for phrase in phrases:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        ids = tokenizer.encode(phrase, add_special_tokens=True)
        # Update the maximum phrase length.
        max_len = max(max_len, len(ids))
        
    #adding 10 extra to max_len just in case
    max_len=max_len+10
    print(max_len)


    # In[17]:


    # Tokenize all of the phrases and map the tokens to their word IDs.
    input_ids = []
    attention_masks = []
    # For every phrase...
    for phrase in phrases:
        # `encode_plus` will:
        #   (1) Tokenize the phrase.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the phrase to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            phrase,                      # phrase to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            truncation = True,
                            max_length = 23 ,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        # Add the encoded phrase to the list.    
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])


    # In[18]:


    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    hidden_states=[]
    with torch.no_grad():
        for i in range(len(input_ids)):
          outputs = model(input_ids[i],attention_mask=attention_masks[i])
          hidden_states.append(outputs[2])


    # In[19]:


    # `hidden_states` has shape [no of sentences x 13 x 1 x max_len x 768]
    phrase_embeddings=[]
    for i in range(len(hidden_states)):
      token_vecs = hidden_states[i][-2][0]
      # `token_vecs` is a tensor with shape [22 x 768]
      # Calculate the average of all token vectors.
      phrase_embedding = torch.mean(token_vecs, dim=0)
      phrase_embeddings.append(phrase_embedding)


    # In[20]:


    from scipy.spatial.distance import cosine

    selected_keyphrases=[]
    similarity=[]
    keyword=[]
    embed=[]

    for i in range(len(phrase_embeddings)):
      sim = 1 - cosine(document_embedding, phrase_embeddings[i])
      similarity.append(sim)
      keyword.append(phrases[i])
      embed.append(phrase_embeddings[i])


    for i in range(len(similarity)):  
      for j in range(0, len(similarity)-i-1):  
        if similarity[j] < similarity[j+1] : 
          similarity[j], similarity[j+1] = similarity[j+1], similarity[j]
          keyword[j], keyword[j+1] = keyword[j+1], keyword[j]
          embed[j], embed[j+1] = embed[j+1], embed[j]

    selected_keyphrases=keyword[0:10]
    selected_keyphrases_similarity=similarity[0:10]
    selected_keyphrases_embeddings=embed[0:10]
    print(selected_keyphrases)

    return render_template('prediction.html', data=selected_keyphrases)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




