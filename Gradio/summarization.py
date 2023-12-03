from openai import OpenAI
client = OpenAI(api_key='')

def topic(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Identify the main idea of the following sentences. Only give me the sentences of the main idea."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5
  )

  return completion.choices[0].message.content

def removal(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Remove any information that is repetitive or not crucial to understanding the main idea and the supporting points."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5
  )

  return completion.choices[0].message.content

def remove_details(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Remove all details such as reasons, facts, examples, or explanations."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5
  )

  return completion.choices[0].message.content

def simple_sent(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Make this sentence simpler without changing sentence meaning."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5
  )

  return completion.choices[0].message.content

def simplify(statement):
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": f"Simplify the following sentences: {statement}",
            }
        ],
        temperature=0.5,
        top_p=0.5,
    )
    return completion.choices[0].message.content

def check_freq(sentence):

  nlp = spacy.load("en_core_web_sm")
  words = [tok.lemma_ for tok in nlp(sentence) if tok.pos_ not in ["PUNCT", "SPACE"]]

  freq_dict = {}
  for word in words:
    freq = word_frequency(word, 'en')
    freq_dict[word] = freq

  vocab = dict(sorted(freq_dict.items(), key=lambda item: item[1]))
  return vocab

def words_exp(sentence):

  freq_dict = check_freq(sentence)

  explain = dict((k, v) for k, v in freq_dict.items() if v < 1e-4)

  words = list(explain.keys())

  return words

def simplify_words(sentence):
  words = words_exp(sentence)
  assis = 'Give me a new verison of sentences which replace these words in simpler synonyms or explanations:'
  for word in words:
    assis += word
    assis += ', '
  assis += 'inside sentences and only give me the new version of explained sentences. Please combine with orginal sentence meanings and keep the original meanings.'

  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
         {"role": "system",
         "content": assis},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )

  return completion.choices[0].message.content

def simplify_structure(sentence):
  completion = client.chat.completions.create(
      model='gpt-4-1106-preview',
      messages=[
        {"role": "system",
         "content": "Simpler sentences' structure, do not give me several points but a coherent paragraph, and not changing original sentences' meanings."},
        {"role": "user",
         "content": sentence}
      ],
      temperature=0.5,
      top_p=0.5,
  )
  return completion.choices[0].message.content

def output(sentence):
    tmp = simplify(sentence)
    tmp = simplify_structure(tmp)
    tmp = simplify_words(tmp)
    tmp = simplify_structure(tmp)
    tmp = simplify_words(tmp)
    return tmp

def summary(sentence):

  tmp = topic(sentence)
  tmp = output(tmp)
  tmp = removal(tmp)
  tmp = remove_details(tmp)
  res = simple_sent(tmp)

  return res
